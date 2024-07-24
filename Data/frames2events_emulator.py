# TODO combine frames2events_ee and frames2events_subp into frames2events and mod datagenerator accordingly

# Passes frame to emulator with step dictated by walker
# Is agnostic to the walk policy 

from configs.envar import CAMERA_RES, IM_SIZE, CIFAR, CIFAR_test
import numpy as np
from v2e.v2ecore.emulator import EventEmulator
import cv2

from typing import Union 

#from PIL import Image 

class StatefulEmulator(EventEmulator):
# self.height, self.width mystery 
    def __init__(self, 
        output_folder = None , dvs_h5 = None,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=0.0,
        num_frames = 300,
        fps = 50,
        pos_thres = 0.4,
        neg_thres = 0.4,
        refractory_period_s=0.0,
        
        im_size = IM_SIZE,
        frame_h = CAMERA_RES[0],
        frame_w = CAMERA_RES[1],
        test_data = False,
        bg_fill = 255 # MOD    
    ):
        
        super().__init__(output_folder=output_folder,
        dvs_h5=dvs_h5,
        leak_rate_hz=leak_rate_hz, 
        shot_noise_rate_hz=shot_noise_rate_hz, 
        refractory_period_s=refractory_period_s, 
        pos_thres=pos_thres, 
        neg_thres=neg_thres, 
        output_height=frame_h,
        output_width=frame_w,
        )

        # self.set_dvs_params("clean")

        # across frames
        self.current_time = 0.
        #self.idx = 0
        
        # not relevant as parameter because one image at a time? 
        #fps = FPS
        self.delta_t = 1 / fps

        # frame times 
        # my understanding of v2e.v2e lines 611 (input_slowmotion_factor = 1.0), 788, 794-797 
        # starting times [0., 0.02,...,5.98] in increments of 0.02 for a total of six seconds
        num_frames += 1 # mod +1
        interpTimes = np.array(range(num_frames)) # mod + 1
        interpTimes = self.delta_t * interpTimes

        self.im_size = (im_size, im_size ) if isinstance(im_size, int) else im_size
        self.frame_hw = (frame_h,frame_w)
        self.center = [self.frame_hw[0]//2 - self.im_size[0]//2, self.frame_hw[1]//2 - self.im_size[1]//2]

        self.img = np.zeros([self.im_size[0],self.im_size[1],3])
        self.luma_img = np.zeros([self.im_size[0],self.im_size[1]])

        # to save other event files 
        self.prepare_storage(num_frames,interpTimes)

        self.data = CIFAR if not test_data else CIFAR_test

        self.fill = bg_fill 
        
# TODO small batch of frames - using n_frames
    def em_frame(self, luma_frame: np.array):

    #    generate_events docstring
    #    """Compute events in new frame.

    #     Parameters
    #     ----------
    #     new_frame: np.ndarray
    #         [height, width], NOTE y is first dimension, like in matlab the column, x is 2nd dimension, i.e. row.
    #     t_frame: float
    #         timestamp of new frame in float seconds

    #     Returns
    #     -------
    #     events: np.ndarray if any events, else None
    #         [N, 4], each row contains [timestamp, x coordinate, y coordinate, sign of event (+1 ON, -1 OFF)].
    #         NOTE x,y, NOT y,x.
    #     """

        new_events = self.generate_events(luma_frame, self.current_time)

        # update timeim2linewalk2events
        self.current_time += self.delta_t

        return new_events

    def reset_(self, img: Union[int,np.array], start_pos: list = None):

        
        # this should not be for restarting the emulator - for the first positioning
        center_coord = self.center if not start_pos else start_pos

        if type(img) == int:
            # self.img = np.array(CIFAR[img][0]) 
            self.img = np.array(self.data[img][0])
        else:
            self.img = img 

        ## equivalent to np.transpose(img,axes=(1,0,2))
        imgT = np.swapaxes(self.img,0,1)
        imgT = np.float32(imgT) 
        # COLOR_BGR2GRAY is a luma conversion of 0.299R+0.587G +0.114B
        # https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html#color_convert_rgb_gray
        if len(imgT.shape) == 3:
            self.luma_img = cv2.cvtColor(imgT, cv2.COLOR_BGR2GRAY) 
        else:
            self.luma_img = imgT 

        frame = np.zeros(self.frame_hw) + self.fill # MOD
        frame[center_coord[0]:center_coord[0]+ self.im_size[0],center_coord[1]:center_coord[1]+self.im_size[1]] = self.luma_img 

        # first frame should return None. If walker_start_pos  
        self.em_frame(luma_frame=frame)

    # img can be the index of a CIFAR img or pass in an img directly (ex to test emulator behavior)
    # coords is either one index to move or a list of coordinates to move
    # walker can be None if pass in coords (such as line)
    def step(self,coords: list = []):

        frame = np.zeros(self.frame_hw)
        # self.luma_img is transposed and greyscaled
        frame[coords[0]:coords[0]+ self.im_size[0],coords[1]:coords[1]+self.im_size[1]] = self.luma_img
        # new events 
        new_events = self.em_frame(luma_frame=frame)

        # return empty list for clean_events aligned with trajectory 
        new_events = [] if new_events is None else new_events

        return new_events
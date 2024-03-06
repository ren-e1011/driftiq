# TODO combine frames2events_ee and frames2events_subp into frames2events and mod datagenerator accordingly

from envar import * 
from v2e.v2ecore.emulator import EventEmulator
import cv2

from typing import Union 

#from PIL import Image 

class StatefulEmulator(EventEmulator):
# self.height, self.width mystery 
    def __init__(self, 
        output_folder = None ,dvs_h5 = None,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=0.0,
        num_frames = 300,
        fps = 50,
        pos_thres = 0.4,
        neg_thres = 0.4,
        refractory_period_s=0.0,
        
        im_size = IM_SIZE,
        frame_h = CAMERA_RES[0],
        frame_w = CAMERA_RES[1]    
    ):
        
        super().__init__(output_folder=output_folder,
        dvs_h5=dvs_h5,
        leak_rate_hz=leak_rate_hz, 
        shot_noise_rate_hz=shot_noise_rate_hz, 
        refractory_period_s=refractory_period_s, 
        pos_thres=pos_thres, 
        neg_thres=neg_thres)

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
        interpTimes = np.array(range(num_frames))
        interpTimes = self.delta_t * interpTimes

        self.im_size = im_size 
        self.frame_hw = (frame_h,frame_w)
        self.img = np.zeros([self.im_size,self.im_size,3])
        self.luma_img = np.zeros([self.im_size,self.im_size])

        # to save other event files 
        self.prepare_storage(num_frames,interpTimes)

        
        
# TODO small batch of frames - using n_frames
    def em_frame(self, luma_frame: np.array):
       
        # from v2e.v2ecore.emulator __main__ tester
        # frame = np.float32(frame)
        # else, passing in a bw frame 
        # luma_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # emulate events
        # self.current_time should be equivalent to interpTimes[i]

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
    
    def _step_init(self,img: Union[int,np.array], walker):
        if type(img) == int:
            self.img = np.array(CIFAR[img][0])
        else:
            self.img = img 
        ## equivalent to np.transpose(img,axes=(1,0,2))
        imgT = np.swapaxes(self.img,0,1)
        imgT = np.float32(imgT) 

        if len(imgT.shape) == 3:
            self.luma_img = cv2.cvtColor(imgT, cv2.COLOR_BGR2GRAY) 
        else:
            self.luma_img = self.img 

        if walker:
            assert walker.sensor_size[0] == self.frame_hw[0]
            assert walker.sensor_size[1] == self.frame_hw[1]

            center_coord = walker.start_pos
        
        else:
            # TODO mod IM_SIZE as in step() im_shape
            center_coord = [self.frame_hw[0]//2 - self.im_size//2, self.frame_hw[1]//2 - self.im_size//2] 

        frame = np.zeros(self.frame_hw)
        frame[center_coord[0]:center_coord[0]+ self.im_size,center_coord[1]:center_coord[1]+self.im_size] = self.luma_img 

        # first frame should return None 
        self.em_frame(luma_frame=frame)

    # img can be the index of a CIFAR img or pass in an img directly (ex to test emulator behavior)
    # coords is either one index to move or a list of coordinates to move
    # walker can be None if pass in coords (such as line)
    def step(self,img: Union[int,np.array], coords: list = [], vec:str = None, walker = None):
        # if this is the first "step" ie img placement 
        if self.img.sum() == 0:
            self._step_init(img,walker)

        assert (len(coords) > 0) or (walker is not None) 
        
        if not coords:
            coords = walker.coord_move(vec)

        frame = np.zeros(self.frame_hw)
        # self.luma_img is transposed and greyscaled
        frame[coords[0]:coords[0]+ self.im_size,coords[1]:coords[1]+self.im_size] = self.luma_img

        # new events 
        new_events = self.em_frame(luma_frame=frame)
        # return empty list for clean_events aligned with trajectory 
        new_events = [] if new_events is None else new_events

        return new_events
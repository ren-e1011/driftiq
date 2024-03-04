# TODO combine frames2events_ee and frames2events_subp into frames2events and mod datagenerator accordingly

from envar import * 
from v2e.v2ecore.emulator import EventEmulator
import cv2

#from PIL import Image 

class StatefulEmulator(EventEmulator):
# self.height, self.width mystery 
    def __init__(self, 
        output_folder = None ,dvs_h5 = None,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=0.0,
        num_frames = 300,
        fps = 50,
        # testing to reduct 
        pos_thres = 0.4,
        neg_thres = 0.4,
        refractory_period_s=0.0
            
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
        # my understanding of v2e.v2e 611 (input_slowmotion_factor = 1.0), 788, 794-797 
        # starting times [0., 0.02,...,5.98] in increments of 0.02 for a total of six seconds
        interpTimes = np.array(range(num_frames))
        interpTimes = self.delta_t * interpTimes

        # to save other event files 
        self.prepare_storage(num_frames,interpTimes)

        
        
# TODO small batch of frames - using n_frames
    def em_frame(self, frame: np.array):
       
        # from v2e.v2ecore.emulator __main__ tester
        frame = np.float32(frame)
        luma_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

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

        # update time
        self.current_time += self.delta_t

        return new_events

    # print event stats
    #    if new_events is not None:
    #        num_events = new_events.shape[0]
    #        start_t = new_events[0, 0]
    #        end_t = new_events[-1, 0]
    #        event_time = (new_events[-1, 0] - new_events[0, 0])
    #        event_rate_kevs = (num_events / self.delta_t) / 1e3

    #        """             print("Number of Events: {}\n"
    #                "Duration: {}\n"
    #                "Start T: {:.5f}\n"
    #                "End T: {:.5f}\n"
    #                "Event Rate: {:.2f}KEV/s".format(
    #            num_events, event_time, start_t, end_t,
    #            event_rate_kevs))
    #        """     

    # def step(self,coords, frame_h = CAMERA_RES[0], frame_w = CAMERA_RES[1], refrac_pd = 0.0, thres=0.4, fps = 50):

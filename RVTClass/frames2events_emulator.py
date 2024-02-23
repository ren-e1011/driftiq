# TODO combine frames2events_ee and frames2events_subp into frames2events and mod datagenerator accordingly

from envar import * 
from v2e.v2ecore.emulator import EventEmulator
import cv2

#from PIL import Image 

class StatefulEmulator(EventEmulator):

    height, width = CAMERA_RES[0], CAMERA_RES[1]

    def __init__(self, 
                 output_folder = None ,dvs_h5 = None,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=0.0,
        output_width = width,
        output_height=height,
        num_frames = 300,
        fps = 50,
        # testing to reduct 
        pos_thres = 0.4,
        neg_thres = 0.4,
        refractory_period_s=0.005
            
    ):
        super().__init__(output_folder=output_folder,dvs_h5=dvs_h5,leak_rate_hz=leak_rate_hz, shot_noise_rate_hz=shot_noise_rate_hz, output_width=output_width, output_height=output_height, refractory_period_s=refractory_period_s, pos_thres=pos_thres, neg_thres=neg_thres)
        # EventEmulator params
        #self.leak_rate_hz = leak_rate_hz
        #self.shot_noise_rate_hz = shot_noise_rate_hz
        #self.output_width = output_width
        #self.output_height = output_height

        # self.set_dvs_params("clean")

        #self.cap = cv2.VideoCapture()

        # across frames
        self.current_time = 0.
        #self.idx = 0
        

        # not relevant as parameter because one image at a time? 
        #fps = args.frame_rate
        self.delta_t = 1 / fps

        # frame times 
        # my understanding of v2e.v2e 611 (input_slowmotion_factor = 1.0), 788, 794-797 
        # starting times [0., 0.02,...,5.98] in increments of 0.02 for a total of six seconds
        interpTimes = np.array(range(num_frames))
        interpTimes = self.delta_t * interpTimes

        # to save other event files 
        self.prepare_storage(num_frames,interpTimes)

        # should save automatically
        #self.events = np.zeros((0,4), dtype=np.float32)
        
        #self.num_frames = num_frames
        #duration = num_of_frames / fps

        #self.step_events_dict = step_events_dict
        # instantiate a new emulator with each image 
        #if imix not in step_events_dict.keys:
    #    self.step_events_dict = {
    #        "coords": [],
    #        "start_t": [],
    #        "end_t": [],
    #        "event_time":[],
    #        "event_rate_k": [],
    #        "step": {}

#        }
        
        
        
# TODO small batch of frames
    def em_frame(self, frame: np.array):

        # if pass in as a numpy array 
        #frame = Image.fromarray(frame)

        # necessary?
        #returnval, frame = cap.read(frame)

        #if returnval:
            # if it fails as an np array can always save to jpg and then load with cap.read() but that feels too inefficient - not array: PIL.Image
        
        # from v2e.v2ecore.emulator __main__ tester
        frame = np.float32(frame)
        luma_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # alternatively, v2e.v2e main() > v2e.v2ecore read_image()
        # seems to be the same in principle but ^ is slightly simpler when feeding in frame x frame 


        # emulate events
        # self.current_time should be equivalent to interpTimes[i]

        #  generate_events(self, new_frame, t_frame):
        # """Compute events in new frame.

        # Parameters
        # ----------
        # new_frame: np.ndarray
        #     [height, width], NOTE y is first dimension, like in matlab the column, x is 2nd dimension, i.e. row.
        # t_frame: float
        #     timestamp of new frame in float seconds

        # Returns
        # -------
        # events: np.ndarray if any events, else None
        #     [N, 4], each row contains [timestamp, x coordinate, y coordinate, sign of event (+1 ON, -1 OFF)].
        #     NOTE x,y, NOT y,x.
        # """

        new_events = self.generate_events(luma_frame, self.current_time)

        #if new_events is not None and new_events.shape[0] > 0:
        #                self.events = np.append(new_events, new_events, axis=0)
        #                self.events = np.array(self.events)

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
    #        self.step_events_dict[self.imix]["coords"].append(step_coords) 
    #        self.step_events_dict[self.imix]["start_t"].append(start_t) 
    #        self.step_events_dict[self.imix]["end_t"].append(end_t)
    #        self.step_events_dict[self.imix]["event_time"].append(event_time)
    #        self.step_events_dict[self.imix]["event_rate_k"].append(event_rate_kevs) 
            # self.idx -> self.frame_counter in emulator. 0-based indexer
    #        self.step_events_dict[self.imix]["step"][self.frame_counter] = new_events         
    

        #self.idx += 1
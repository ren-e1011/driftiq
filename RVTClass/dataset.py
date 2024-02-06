from envar import *

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from datagenerator import im2frames2events
from preprocess import construct_x

class DataSet(Dataset):

    def __init__(self, args):
        self.walk = args.walk
        self.events_path = RAND_EVENTSDIR if self.walk == 'random' else INFO_EVENTSDIR 

        self.nsteps = args.n_frames
        self.fps = args.frame_rate_hz

    def get_events(self,i):
        # 300 frames at 50fps is 6 seconds - with 0.02s time steps
        # biological drift speed ~40 arcmin/second for ~300ms 

        # create events if does not exist

        # if not os.path.isfile(self.events_path+f"/Im_{i}.h5"):
        #     im2frames2events(i, walk = self.walk, nsteps = self.nsteps, fps = self.fps )
        events = None
        try: 
            events = np.array(h5py.File(self.events_path+f"/Im_{i}.h5")['events'], dtype='int32')
            # whether file does not exist or is corrupted
            # OSError or FileNotFoundError 
        except OSError:
            if os.path.isfile(self.events_path+f"/Im_{i}.h5"):
                os.remove(self.events_path+f"/Im_{i}.h5")
            im2frames2events(i, walk = self.walk, nsteps = self.nsteps, fps = self.fps )
            events = np.array(h5py.File(self.events_path+f"/Im_{i}.h5")['events'], dtype='int32')
            
        
        return events


    def __len__(self):
        # returns 3125 - 50k/batch_size=16
        return len(CIFAR) 

    def __getitem__(self,index):
        # t_getx_start = t_getsample_start = time.time()
        x = self.get_events(index)
        
        # y is a torch.uint8

        _,y = CIFAR[index]
        

        # to retain [19] not a tensor with shape 19 
        # y = torch.cuda.IntTensor([y])
        y = torch.tensor([y], dtype=torch.int)
        # y = y.type_as(y)

# TODO testing uniform len pre-streaming 
        # 16/1600 (0.01) - noise events - which cut over the CAMERA_RES limit for computations
        # at time stamps (/1600) [ 46,  54, 109, 162, 163, 176, 227, 262, 337, 350, 357, 392, 421, 502, 675, 705]
        # Noise events confirmed by confirming that the im2randomwalk.randTraj2Frames walk stays within the CAMERA_RES boundaries
        # Noise seems to increase with filming duration (n01 as total recording had .36 events cross the - height - threshold but first 1600 events had 0.01)
        # 698111/1105756 noiseless events
        # all violators are in the height...? camera_res reduces dvs_res in height more than width but still...
        # suspect of bug
        x = x[(x[:,1] < CAMERA_RES[0]) & (x[:,2] < CAMERA_RES[1])]
        # shouldnt need this with args.height, width
        
        x = x[:SEQ_LEN,:]
        # mod [x] -> x
        # UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor
        # x = torch.cuda.IntTensor([x])
        # mod rm self.device 
        # x = torch.tensor([x], dtype=torch.int)
        x = torch.tensor(x).unsqueeze(dim=0)
        # x = x.type_as(x)

        x = construct_x(x)
        # t_getx_end = t_getsample_end = time.time()

        # t_sample_load = t_getsample_end - t_getsample_start
        # t_x_load = t_getx_end - t_getx_start
        # print(f"time to load sample {index}", t_sample_load)
        # print('time to load x', t_x_load)
        # print('time to load y', t_sample_load - t_x_load)
        return x,y

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--walk", type=str, default='random')

    parser.add_argument("--n_frames",type=int,default=300)
    parser.add_argument("--frame_rate_hz",type=int,default=50)

    # run list of indices in parallel
    # number of data to generate
    parser.add_argument("--n_im", type=int, default=4)
    # Clean vs noisy for events generation - not relevant for this project
    parser.add_argument("--condition",type=str,default="Clean")


    args = parser.parse_args()
    ds = DataSet(args)
    x,y = ds.__getitem__(index=2)

from envar import *

import os
import pickle
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from omegaconf import DictConfig
from datagenerator import im2frames2events, im2krandsteps2events
from preprocess import construct_x, cutEdges, cutEdges_savedData
import itertools

from frames2events_emulator import StatefulEmulator
from im2randomwalk import RandomWalk

class DataSet(Dataset):

    def __init__(self, config: DictConfig, args):
        self.config = config
        self.walk = args.walk
        self.events_path = RAND_EVENTSDIR if self.walk == 'random' else INFO_EVENTSDIR 
        self.traj_path = RAND_TRAJPATH if self.walk == 'random' else INFO_TRAJPATH

        self.nsteps = args.n_frames
        self.k = args.timesteps
        self.rp = args.refrac_pd
        self.thres = args.threshold

        self.fps = args.frame_rate_hz
        self.generate_k_timesteps = not args.use_saved_data 

        self.j_timebins = self.config.model.backbone.input_channels

    # if frames have been 
    def get_events(self,i):
        # 300 frames at 50fps is 6 seconds - with 0.02s time steps
        # biological drift speed ~40 arcmin/second for ~300ms 

        # create events if does not exist
        events = None
        imtraj_pkl = os.path.join(self.traj_path,f"Im_{i}.pkl")
            
        try: 
            events = np.array(h5py.File(self.events_path+f"/Im_{i}.h5")['events'], dtype='int32')
            with open(imtraj_pkl, 'rb') as fp:
            # dict of event counts accessed by [x][y]
                imtraj = pickle.load(fp)
            # whether file does not exist or is corrupted
            # OSError or FileNotFoundError 
        except OSError:
            if os.path.isfile(self.events_path+f"/Im_{i}.h5"):
                os.remove(self.events_path+f"/Im_{i}.h5")

            if os.path.isfile(self.traj_path+f"/Im_{i}.pkl"):
                os.remove(self.traj_path+f"/Im_{i}.pkl")

            im2frames2events(i, walk = self.walk, nsteps = self.nsteps, fps = self.fps )

            events = np.array(h5py.File(self.events_path+f"/Im_{i}.h5")['events'], dtype='int32')
            with open(imtraj_pkl, 'rb') as fp:
            # dict of event counts accessed by [x][y]
                imtraj = pickle.load(fp)
            
        
        return events, imtraj
    
    def get_k_events(self,i,timesteps=40,refrac_pd = 0.0, thres=0.4, start_pos: list = []):

        if self.walk == 'random': 
            # events is a list of 
            events, imtraj = im2krandsteps2events(i=i, timesteps=timesteps, refrac_pd=refrac_pd, thres=thres, fps = self.fps, start_pos = start_pos)

            # even if zero
            # spikes_count.append(num_events)
        else:
            raise NotImplementedError

        return events, imtraj 


    def __len__(self):
        # returns 3125 - 50k/batch_size=16
        return len(CIFAR) 

    
    
    def preprocess(self, x, traj,index):
        
        if self.generate_k_timesteps:
            x, _ = cutEdges(x, traj)
        
        else:
            x = cutEdges_savedData(x, traj)
        
        x = np.concatenate(x, axis=0)
        x = torch.tensor(x)
        # violations outside of sensor space? shouldnt be an issue any more 
        # x = x[(x[:,1] < CAMERA_RES[0]) & (x[:,2] < CAMERA_RES[1])]
        # x = construct_x(x, t_steps = self.config.model.backbone.input_channels, step_t = 1/self.fps,t_res= self.config.model.backbone.input_channels)
        # nbins is defined in envar but also aligns with self.config.model.backbone.input_channels to avoid runtimeerror - see definition in envar.py
        x = construct_x(x, step_t = 1/FPS, bins = self.j_timebins)

        
        return x

    def __getitem__(self,index):

        # t_getx_start = t_getsample_start = time.time()
        # self.k_events returns a list of arrays of len(timestamps), each of n x 4 dimensions
        events, traj = self.get_k_events(index, timesteps = self.k, refrac_pd=self.rp, thres=self.thres) if self.generate_k_timesteps else self.get_events(index)
        x = self.preprocess(events, traj,index)
        
        # y is a torch.uint8
        _,y = CIFAR[index]
        

        # to retain [19] not a tensor with shape 19 
        y = torch.tensor([y], dtype=torch.int)

# TODO testing uniform len pre-streaming 
        # 16/1600 (0.01) - noise events - which cut over the CAMERA_RES limit for computations
        # at time stamps (/1600) [ 46,  54, 109, 162, 163, 176, 227, 262, 337, 350, 357, 392, 421, 502, 675, 705]
        # Noise events confirmed by confirming that the im2randomwalk.randTraj2Frames walk stays within the CAMERA_RES boundaries
        # Noise seems to increase with filming duration (n01 as total recording had .36 events cross the - height - threshold but first 1600 events had 0.01)
        # 698111/1105756 noiseless events
        # all violators are in the height...? camera_res reduces dvs_res in height more than width but still...
        
        # MOD time instead of steps
        # x = x[:SEQ_LEN,:]


        # mod [x] -> x
        # UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor
        # x = torch.cuda.IntTensor([x])
        # mod rm self.device 
        # x = torch.tensor([x], dtype=torch.int)
        
        # x = x.type_as(x)

        # x = torch.tensor(x).unsqueeze(dim=0)
        
        return x,y

from omegaconf import DictConfig, OmegaConf
if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--walk", type=str, default='random')

    parser.add_argument("--n_frames",type=int,default=300)
    parser.add_argument("--frame_rate_hz",type=int,default=50)
    
    parser.add_argument("--use_saved_data", type=bool, default = False)
    parser.add_argument("--timesteps", type=int, default=40)
    parser.add_argument("--refrac_pd", type=int, default=0.0)
    parser.add_argument("--threshold", type=int, default=0.4)



    args = parser.parse_args()
    config = OmegaConf.load('./RVTClass/config/train.yaml')
    ds = DataSet(config, args)
    x,y = ds.__getitem__(index=2)

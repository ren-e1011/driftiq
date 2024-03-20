from envar import *

import os
import pickle
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from datagenerator import im2events
from preprocess import construct_x, cutEdges

from frames2events_emulator import StatefulEmulator


class DataSet(Dataset):

    def __init__(self, args, steps, bins):
        self.walk = args.walk
        self.events_path = RAND_EVENTSDIR if self.walk == 'random' else INFO_EVENTSDIR 
        self.traj_path = RAND_TRAJPATH if self.walk == 'random' else INFO_TRAJPATH

        # number of timesteps
        self.n_steps = steps
        # number of timebins 
        self.j_timebins = bins

        self.rp = args.refrac_pd
        self.thres = args.threshold
        self.fps = args.frame_rate_hz

        self.use_saved = args.use_saved_data
        # CAMERA_RES
        self.frame_h = args.frame_hw[0]
        self.frame_w = args.frame_hw[1]

        

    # if frames have been saved into h5 file timesteps
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

            im2events(i, walk = self.walk, nsteps = self.n_steps, fps = self.fps, save = True )

            events = np.array(h5py.File(self.events_path+f"/Im_{i}.h5")['events'], dtype='int32')
            with open(imtraj_pkl, 'rb') as fp:
            # dict of event counts accessed by [x][y]
                imtraj = pickle.load(fp)
            
        
        return events, imtraj
    
    def get_k_events(self,i,start_pos: list = [],k_steps: int = None):

        k_steps = self.n_steps if not k_steps else k_steps 

        # cutEdges is implicit in im2events 
        events, nevents, imtraj = im2events(img=i, walk = self.walk, nsteps=k_steps, 
                                   pos_thres=self.thres,neg_thres=self.thres, 
                                   refrac_pd=self.rp,fps=self.fps, 
                                   # to restart walk 
                                   start_pos = start_pos, 
                                    frame_h = self.frame_h, frame_w = self.frame_w)
        
        return events, imtraj 

    def __len__(self):
        # returns 3125 - 50k/batch_size=16
        return len(CIFAR) 

    
    
    def preprocess(self, x, traj,index):
        
        # currently 19.3.24 cutEdges is subsumed in im2events for info walk. (re)consider if dataset is presaved  
        # if not self.use_saved:
        #     x, _ = cutEdges(x, traj)
        
        # else:
        #     x, _ = cutEdges(x, traj, from_saved=True,steps = self.n_steps)
        
        # to collapse list of timesteps 
        # to be swapped out with architecture
        x = np.concatenate(x, axis=0)
        x = torch.tensor(x)

        x = construct_x(x, step_t = 1/FPS, bins = self.j_timebins, height = self.frame_h, width = self.frame_w)

        
        return x

    def __getitem__(self,index):

        # t_getx_start = t_getsample_start = time.time()
        # self.k_events returns a list of arrays of len(timestamps), each of n x 4 dimensions
        events, traj = self.get_k_events(index) if not self.use_saved else self.get_events(index)

        x = self.preprocess(events, traj,index)
        
        # y is a torch.uint8
        _,y = CIFAR[index]
        # to retain [19] not a tensor with shape 19 
        y = torch.tensor([y], dtype=torch.int)

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

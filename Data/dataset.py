from pathlib import Path 
import os, sys
sys.path.append(str(Path.cwd().parent)) # for pythonpath 
sys.path.append(str(Path.cwd()))

# from configs.envar import RAND_EVENTSDIR, INFO_EVENTSDIR, TS_EVENTSDIR, RAND_TRAJPATH, INFO_TRAJPATH, TS_TRAJPATH, CIFAR, CIFAR_test
# from configs.envar import RAND_EVENTSDIR, INFO_EVENTSDIR, TS_EVENTSDIR, RAND_TRAJPATH, INFO_TRAJPATH, TS_TRAJPATH, FILEPATH

import os
import pickle
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from Data.datagenerator import im2events
from utils.preprocess import construct_x


from torch.nn.utils.rnn import pad_sequence

class Collator(object):
    '''
    Yields a batch from a list of Items
    Args:
    test : Set True when using with test data loader. Defaults to False
    percentile : Trim sequences by this percentile
    '''
    # test parameter is for validation** Not test which has ys 
    def __init__(self,test=False,percentile=100):
        self.test = test
        # self.percentile = percentile # not important for our use case where we are going to lob off all padding using returned lengths
    def __call__(self, batch):
        if not self.test:
            x, lengths, y = zip(*batch)
            # lengths is a nested list of events for each timestep in each sample
            sample_lengths = [sum(lens) for lens in lengths]
            # x = [item[0] for item in batch]
            # lengths = [item[1] for item in batch]
            # y = [item[2] for item in batch]
        else:
            x, lengths = zip(*batch)
            sample_lengths = [sum(lens) for lens in lengths]
            # x = [item[0] for item in batch]
            # lengths = [item[1] for item in batch]

        # max_len = max(sample_lengths)
        # lens = [len(x) for x in data]
        # max_len = np.percentile(lens,self.percentile)
        x = pad_sequence(x,batch_first=True) # automatically pads to max_len
        # x = torch.tensor(x,dtype=torch.long) 
        if not self.test:
            y = torch.tensor(y,dtype=torch.int) # float32 to int
            return [x,lengths, y]
        return [x, lengths]



class DataSet(Dataset):

    def __init__(self, architecture, walk, img_dataset,
                 steps, bins, refrac_pd, threshold, use_saved_data, paths,
                 frame_hw, fps, preproc_data = True,ts_s = 50, ts_mu = 500, test=False):
        self.architecture = architecture
        self.walk = walk
        
        walk_dir = paths.ts_walk if self.walk == 'ts' else paths.rand_walk # other walks implemented need to be added here or if other walks in use such as 
        test_dir = 'Test' if test else ''

        self.events_path = os.path.join(paths.filepath, walk_dir, test_dir, "Events")
        self.traj_path = os.path.join(paths.filepath,walk_dir,test_dir,"Trajectories")
        self.hits_path = os.path.join(paths.filepath, walk_dir,test_dir,"HitLists")

        # number of timesteps
        self.n_steps = steps
        # number of timebins 
        self.j_timebins = bins

        self.rp = refrac_pd
        self.thres = threshold
        self.fps = fps
        self.ts_sigma = ts_s
        self.ts_mu = ts_mu

        # self.data = CIFAR if not test else CIFAR_test
        self.data = img_dataset
        self.test_data = test
        self.use_saved = use_saved_data
        self.paths = paths

        # CAMERA_RES
        self.frame_h = frame_hw[0]
        self.frame_w = frame_hw[1]

        self.preproc_data = preproc_data


    # def events_by_step(events, nevents):
    #     events_

    # Test 
    def get_saved_events(self,i):
        hits_pkl = os.path.join(self.hits_path,f"Im_{i}.pkl")
        imtraj_pkl = os.path.join(self.traj_path,f"Im_{i}.pkl")
        try: 
            events = np.array(h5py.File(self.events_path+f"/Im_{i}.h5")['events'], dtype='int32')

            
            with open(hits_pkl, "rb") as file:
                nevents = pickle.load(file)
            # nevents = np.load(self.events_path+f"/Im_{i}.h5", allow_pickle=True)
            # nevents = [len(e) for e in events]
            
            with open(imtraj_pkl, 'rb') as fp:
            # dict of event counts accessed by [x][y]
                imtraj = pickle.load(fp)  
                
        except OSError:
            print("One or more files of im {i} does not exist or is corrupted. Generating new trajectory")
            events, nevents, imtraj = self.get_k_events(i, save = True)
            # TODO save 
        return events, nevents, imtraj
    
    
    def get_k_events(self,i,start_pos: list = [],k_steps: int = None, save = False):

        k_steps = self.n_steps if not k_steps else k_steps 

        # cutEdges is implicit in im2events 
        events, nevents, imtraj = im2events(img=np.array(self.data[i][0]), imix=i, walk = self.walk, nsteps=k_steps, 
                                   pos_thres=self.thres,neg_thres=self.thres, 
                                   refrac_pd=self.rp,fps=self.fps,
                                   paths = self.paths, 
                                   # to restart walk 
                                   start_pos = start_pos, 
                                   frame_h = self.frame_h, frame_w = self.frame_w, preprocess= self.preproc_data,
                                   ts_mu= self.ts_mu, ts_w = self.ts_sigma,
                                   test_data=self.test_data, save= save)
        
        events = np.concatenate(events,axis=0)
        # TODO verify nevents => length in /home/renaj/Driftiq/matrixlstm/classification/libs/trainer.py batch_lengths for batch in dataloader
        return events, nevents, imtraj 

    def __len__(self):
        # returns 3125 - 50k/batch_size=16
        return len(self.data) 

    
    
    def preprocess(self, x, traj,index):
        
        # currently 19.3.24 cutEdges is subsumed in im2events for info walk. (re)consider if dataset is presaved  
        # if not self.use_saved:
        #     x, _ = cutEdges(x, traj)
        
        # else:
        #     x, _ = cutEdges(x, traj, from_saved=True,steps = self.n_steps)
        
        # to collapse list of timesteps 

        # to be swapped out with architecture

        # TODO 
        # if len(x) > self.n_steps: #if saved, concatenated
        #     x = np.concatenate(x, axis=0)

        x = torch.tensor(x)

        if self.architecture == 'rvt':
            to_frame = True
            
            x = construct_x(x, to_frame = to_frame, step_t = 1/self.fps, bins = self.j_timebins, height = self.frame_h, width = self.frame_w)

        
        return x
    
    

    def __getitem__(self,index):

        # t_getx_start = t_getsample_start = time.time()
        # self.k_events returns a list of arrays of len(timestamps), each of n x 4 dimensions
        events, nevents, traj = self.get_k_events(index) if not self.use_saved else self.get_saved_events(index)

        x = self.preprocess(events, traj,index) # different preprocessing for mxlstm
        
        # y is a torch.uint8
        _,y = self.data[index]
        # to retain [19] not a tensor with shape 19 
        y = torch.tensor(y, dtype=torch.int) # mod? 

        return x, nevents, y

from omegaconf import DictConfig, OmegaConf
from torchvision import datasets
import argparse 
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--walk", type=str, default='random')

    parser.add_argument("--n_frames",type=int,default=300)
    parser.add_argument("--frame_rate_hz",type=int,default=50)
    
    parser.add_argument("--use_saved_data", type=bool, default = False)
    parser.add_argument("--timesteps", type=int, default=40)
    parser.add_argument("--refrac_pd", type=int, default=0.0)
    parser.add_argument("--threshold", type=int, default=0.4)

    dataset = datasets.MNIST(os.path.join(FILEPATH,'SavedData/'),train=True,download=True, transform=None)
    # dataset = datasets.CIFAR(os.path.join(FILEPATH,'SavedData/'),train=True,download=True, transform=None)

    args = parser.parse_args()
    config = OmegaConf.load('configs/mxlstm_cfg.yaml')
    # print(config['model']['name'])
    # print(args.walk)
    # ds = DataSet(config, args)
    ds = DataSet(config['model']['name'], args.walk, dataset,
                 steps=args.timesteps, bins=40, refrac_pd=args.refrac_pd, threshold=args.threshold, use_saved_data=args.use_saved_data, 
                 frame_hw=config['emulator']['frame_hw'], fps=50, preproc_data = False,ts_s = 50, ts_mu = 500, test=False)
    x,y,z = ds.__getitem__(index=2)

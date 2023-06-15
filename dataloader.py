from envar import *
from datagenerator import generate_randomwalk_events

from torch.utils.data import Dataset
import argparse
from subprocess import Popen

import os
from random import sample

from frames2events_subp import im2frames2events

import h5py

from torch.cuda import IntTensor

# move to run file
parser = argparse.ArgumentParser()

parser.add_argument("--data_path",type=str,default="./Data")
parser.add_argument("--video_path", type=str, default="./Data/RandomImWalk/300Frames50Hz")
parser.add_argument("--save_walk_video",type=bool, default=True)

parser.add_argument("--events_path", type=str, default="./Data/RandomImWalk/346DVSEvents")
parser.add_argument("--camera_config",type=str,default="dvs346")
# TODO after testing mod to default=None 
parser.add_argument("--traj_path", type=str,default="./Data/RandomImWalk/Trajectories")
parser.add_argument("--walk", type=str, default='random')

parser.add_argument("--n_frames",type=int,default=300)
parser.add_argument("--frame_rate_hz",type=int,default=50)

# run list of indices in parallel
# number of data to generate
parser.add_argument("--n_im", type=int, default=4)
# Clean vs noisy for events generation - not relevant for this project
parser.add_argument("--condition",type=str,default="Clean")

args = parser.parse_args()



class DataLoader(Dataset):

    def __init__(self, events_path = "./Data/RandomImWalk/346DVSEvents"):

            self.events_path = events_path
            

    def get_events(self,i):
        # 900 frames at 30fps is 30 seconds - ?
        # drift speed ~40 arcmin/second - 
        # don't record all the steps 
        # im_ixs = [i for i in range(len(DATASET))]
        # random.sample - without replacement (random.choices with replacement)
        # ixs = sample(im_ixs,args.n_im)
        
        # create events if does not exist

        if not os.path.isfile(self.events_path+f"/Im_{i}.avi") and args.walk == 'random':
            generate_randomwalk_events(i)

        events = np.array(h5py.File(self.events_path+f"/Im_{i}.h5")['events'], dtype='int32')

        # int tensor
        return IntTensor(events)


    def __len__(self):

        return len(CIFAR)

    def __getitem__(self,index):

        x = self.get_events(index)

        # y is a torch.uint8
        _,y = CIFAR[index]
        
        y = IntTensor(np.array([y],dtype='int32'))

        return x,y

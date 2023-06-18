from envar import *
# from subprocess import Popen

import os
# from random import sample

from torch.utils.data import Dataset
from torch.cuda import IntTensor

from datagenerator import generate_randomwalk_events
from frames2events_subp import im2frames2events

import h5py

import numpy as np



class DataSet(Dataset):

    def __init__(self, args):

            self.events_path = args.events_path
            self.walk = args.walk
            

    def get_events(self,i):
        # 900 frames at 30fps is 30 seconds - ?
        # drift speed ~40 arcmin/second - 
        # don't record all the steps 
        # im_ixs = [i for i in range(len(DATASET))]
        # random.sample - without replacement (random.choices with replacement)
        # ixs = sample(im_ixs,args.n_im)
        
        # create events if does not exist

        if not os.path.isfile(self.events_path+f"/Im_{i}.avi") and self.walk == 'random':
            generate_randomwalk_events(i)

        elif not os.path.isfile(self.events_path+f"/Im_{i}.avi") and self.walk == 'info':
            raise NotImplementedError


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

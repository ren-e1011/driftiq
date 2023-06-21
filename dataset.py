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

# mod /home/renee/DIQ/RVT/data/genx_utils/dataset_streaming.py > build_streaming_dataset
from typing import List, Union
# mod from RVTClass.data.cifar_utils.sequence_for_streaming > SequenceForIter
from RVTClass.data.utils.stream_concat_datapipe import ConcatStreamingDataPipe
from RVTClass.data.cifar_utils.sequence_base import SequenceBase
from pathlib import Path
from typing import List, Optional, Union, Tuple
# mod from /home/renee/DIQ/RVT/data/genx_utils/sequence_base.py
import torch

# SequenceForIter.get_sequences_with_guaranteed_labels(
#             path=path,
#             ev_representation_name=ev_representation_name,
#             sequence_length=sequence_length,
#             dataset_type=dataset_type,
#             downsample_by_factor_2=downsample_by_factor_2)


class DataSet(Dataset):

    def __init__(self, args):

            self.events_path = args.events_path
            self.walk = args.walk
            

    def get_events(self,i,zero_offset=True):
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

        # TODO revisit datatypes if training fails
        # if ev_repr.dtype != torch.uint8:
            # ev_repr = torch.asarray(ev_repr, dtype=torch.float32)
            # ^ from RVT/data/genx_utils/sequence_base.py > SequenceBase _get_event_repr_torch
        events = np.array(h5py.File(self.events_path+f"/Im_{i}.h5")['events'], dtype='int32')
        
        # Does recording start at 0 and first event is first event or is there some alternative internal clock 
        if zero_offset:
            # set time column to start at 0 and insert as first column  
            events = np.insert(events,0,np.array(events[:,0]-events[0,0],axis=1))
            # remove original time column
            events = np.delete(events,1,axis=1)
            
        # int tensor - TODO revisit datatype if training fails
        return IntTensor(events)
    # parameters from config.dataset // RVT/config/dataset > base.yaml
# // RVT/config > general.yaml
# arbitrary sequence_length
# def build_streaming_dataset(dataset_mode='train', batch_size = 8, num_workers = 6, sequence_length = 8)
    # RVT/data/genx_utils/sequence_base.py > SequenceBase
    



    def __len__(self):
        # TODO revisit whether sequence length impacts datastream length 
        return len(CIFAR) 

    def __getitem__(self,index):

        x = self.get_events(index)

        # y is a torch.uint8
        _,y = CIFAR[index]
        
        y = IntTensor(np.array([y],dtype='int32'))

        return x,y

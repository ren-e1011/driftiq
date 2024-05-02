import os
 # try 

# which of these are necessary or useful 
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# from RVT > train.py
# see https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES']='2, 3'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# https://freedium.cfd/https://iamholumeedey007.medium.com/memory-management-using-pytorch-cuda-alloc-conf-dabe7adec130
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1"

import numpy as np
from torchvision import datasets

FILEPATH = '/home/renaj/DIQ'
RAND_ = "SavedData/RandomImWalk"
INFO_ = "SavedData/InfoImWalk"

# TODO rm
RVT_FILEPATH = '/home/renaj/DIQ/RVTClass'
OUTDIR = './Data/RandomImWalk/300Frames50Hz/'

# home filepath (DIQ) + 
RAND_TRAJPATH = os.path.join(FILEPATH, RAND_ ,"Trajectories")
INFO_TRAJPATH = os.path.join (FILEPATH, INFO_, "Trajectories")
# TODO mod
RAND_EVENTSDIR = os.path.join(FILEPATH, RAND_, "Events")
INFO_EVENTSDIR = os.path.join(FILEPATH, INFO_, "Events")

RAND_HITSDIR = os.path.join(FILEPATH, RAND_, "HitLists")
INFO_HITSDIR = os.path.join(FILEPATH, INFO_, "HitLists")

# should be unnecessary with pytorch_lightning
# USE_CUDA = True
# if USE_CUDA:
#     torch.set_default_device('cuda')

# failed effort to use multiple workers in train - throws RuntimeError: context has already been set
# torch.multiprocessing.set_start_method('spawn')
# swapped 346, 260 by inspection...shouldnt matter
# https://inivation.com/wp-content/uploads/2019/08/DAVIS346.pdf
# Width: 346 pixels x 18.5 um/pixel = 6.4 mm
# Height: 260 pixels x 18.5 um/pixel = 4.81 mm
# also copied to /home/renee/DIQ/RVTClass/config/dataset/cifar.yaml resolution_hw
# crop camera res to gen1 resolution (240,304) for parameter consistency see RVT.config.datasets > gen1.yaml and gen1 paper https://arxiv.org/pdf/2001.08499.pdf
# factual
# CAMERA_RES = (260,346,3)
# gen1
# CAMERA_RES = (240,304,3)
# height, width 
DVS_RES = (260,346,3)
# fictional 
# CAMERA_RES = (192,320,3)
CAMERA_RES = (96,96,3)
# CAMERA_RES = DVS_RES

# default SENSOR.dtype is dtype('int64')
# SENSOR = np.zeros(CAMERA_RES,dtype=int)

CIFAR = datasets.CIFAR100(os.path.join(FILEPATH,'SavedData/'),train=True,download=True, transform=None)
CIFAR_test = datasets.CIFAR100(os.path.join(FILEPATH,'SavedData/'),train=False,download=True, transform=None)
# 32
IM_SIZE = CIFAR[0][0].size[0]
N_CLASSES = 100 
# mod camera_res to dvs_res? 
CENTER = [CAMERA_RES[0]//2 - IM_SIZE//2, CAMERA_RES[1]//2 - IM_SIZE//2]
# determines time step 
FPS = 50

# 42500 .85*50k
TRAIN_SPLIT_SZ = .85
VAL_SPLIT_SZ = 1 - TRAIN_SPLIT_SZ

# if self.train else downloaded_list = self.test_list - but there is a test folder?
# TESTSET = DATASET = datasets.CIFAR100(os.path.join(FILEPATH,'Data/'),train=False,download=True, transform=None)

# mod from 16
BATCH_SIZE = 32
# for testing
EPOCHS = 50


SEQ_LEN = 1600 # number of events, not length of video 
# from RVT/scripts/genx/preprocess_dataset.py > process_sequence 
ts_step_ev_repr_ms = 50

height = CAMERA_RES[0]
width = CAMERA_RES[1]

EPSILON = 1e-6

# and config file variables from conf_preprocess/representation/stacked_hist.yaml,conf_preprocess/extraction/const_duration.yaml called by RVT/scripts/genx/README.md > For the Gen1 dataset ''' '''
NUM_PROCESSES = 20
ev_repr_num_events = 50 # if config.event_window_extraction.method == AggregationType.COUNT in RVT/scripts/genx/conf_preprocess/extraction/const_duration.yaml (method:DURATION)
ev_repr_delta_ts_ms = 50 #  if config.event_window_extraction.method == AggregationType.DURATION in RVT/scripts/genx/conf_preprocess/extraction/const_duration.yaml (method:DURATION)

# RVT/scripts/genx/preprocess_dataset.py > labels_and_ev_repr_timestamps
ts_step_frame_ms = 100

# RuntimeError: Given groups=1, weight of size [64, 20, 7, 7], expected input[2, 10, 260, 346] to have 20 channels, but got 10 channels instead:: nbins (T) is set to 10 but because they flatten the polarity which is delivered as 2 channels in theirs and 1 in ours, presume nbins should be 20 
# ^ 10 bins for each channel? 
nbins = 20 # T 
count_cutoff = 10 #...or MISSING?

#### Unnecessary 
# preprocess_dataset > process_sequence 
align_t_ms = 100 
# preprocess_dataset > process_sequence > labels_and_ev_repr_timestamps
# 
# mu-seconds 
align_t_us = align_t_ms * 1000
delta_t_us = ts_step_ev_repr_ms * 1000

COMET_API_KEY = ''
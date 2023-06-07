import argparse
from subprocess import Popen

import os
import pickle
from copy import deepcopy
from random import shuffle, sample, choice

import pandas as pd
import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.io import write_video
import cv2

from matplotlib import pyplot as plt

from frames2events_subp import im2frames2events


FILEPATH = '/home/renee/DIQ'
os.chdir(FILEPATH)
OUTDIR = './Data/RandomImWalk_900Frames30Hz/'
# swapped 346, 260 by inspection...shouldnt matter
CAMERA_RES = (260,346,3)
# default SENSOR.dtype is dtype('int64')
SENSOR = np.zeros(CAMERA_RES,dtype=int)

DATASET = datasets.CIFAR100(os.path.join(FILEPATH,'Data/'),train=True,download=True, transform=None)
IM_SIZE = DATASET[0][0].size[0]


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

parser.add_argument("--condition",type=str,default="Clean")

args = parser.parse_args()


args.dataset = DATASET

if __name__ == '__main__':
    # 900 frames at 30fps is 30 seconds - ?
    # drift speed ~40 arcmin/second - 
    # don't record all the steps 
    im_ixs = [i for i in range(len(DATASET))]
    # random.sample - without replacement (random.choices with replacement)
    ixs = sample(im_ixs,args.n_im)
    

    # mod to change 
    # im2frames2events returns the command
    procs = [Popen(im2frames2events(args,i)) for i in ixs]

    for p in procs:
        p.wait()
        
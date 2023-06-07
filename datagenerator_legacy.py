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

import frames2events


FILEPATH = '/home/renaj/DIQ'
os.chdir(FILEPATH)
OUTDIR = './Data/RandomImWalk_900Frames30Hz/'
# swapped 346, 260 by inspection...shouldnt matter
CAMERA_RES = (260,346,3)
# default SENSOR.dtype is dtype('int64')
SENSOR = np.zeros(CAMERA_RES,dtype=int)

DATASET = datasets.CIFAR100(os.path.join(FILEPATH,'Data/'),train=True,download=True, transform=None)
IM_SIZE = DATASET[0][0].size[0]

# in principle could process the image as a lawnmower
# n_steps OUT_DIM**2 for lawnmower is overkill - find some basis/justification for step count. works on full set. trial on small amount for testing
# choose some step count in which probabilistically all <24> *patches will be traversed moving pixelwise -- considering global attention is patch-by-patch
def imgRandomWalk(n_steps = CAMERA_RES[0] * CAMERA_RES[1], size = CAMERA_RES):
    size = (size,size) if isinstance(size, int) else size
    # NW (upper left) coordinate
    # first dimension height. second dimension width
    init_tup = [choice(range(size[0])),choice(range(size[1]))]
    # for now initialize them all uniformly so network does not learn placement:class "association"? but there are many of the same class and sparse representation should take care of that?
    # init_tup = [size[0]//2,size[1]//2]
    walk = []
    # cur = init_tup
    cur = deepcopy(init_tup)
    # print('init_tup',cur)
    # starting point is governed by saccade - collect following
    # walk.append(cur)
    
    directions = ['N','E','S','W']
    edge  = True
    
    for i in range(n_steps):
        temp = deepcopy(cur)
        while edge: 
            step = choice(['N','E','S','W', 'NE', 'NW', 'SE', 'SW'])
            # print('step direction',step)
            if ((step == 'N') and (temp[0] - 1 >= 0)):
                # print('N Step')
                temp[0] -= 1
                edge = False
                
            elif ((step == 'NE') and (temp[0] - 1 >= 0) and (temp[1] + 1 < size[1])):
                # print('NE Step')
                temp[0] -= 1
                temp[1] += 1
                edge = False
                
            elif ((step == 'E') and (temp[1] + 1 < size[1])):
                # print('E Step')
                temp[1] += 1
                edge = False 
                
            elif ((step == 'SE') and (temp[0] + 1 < size[0]) and (temp[1] + 1 < size[1])):
                temp[0] += 1
                temp[1] += 1
                edge = False 
                
            elif ((step == 'S') and (temp[0] + 1 < size[0])):
                # print('S Step')
                temp[0] += 1
                edge = False
                
            elif ((step == 'SW') and (temp[0] + 1 < size[0]) and (temp[1] - 1 >= 0)):
                # print('SW step')
                temp[0] += 1
                temp[1] -= 1
                edge = False

            elif ((step == 'W') and (temp[1] - 1 >= 0)):
                # print('W Step')
                temp[1] -= 1
                edge = False
                
            elif ((step == 'NW') and (temp[0] - 1 >= 0) and (temp[1] - 1 >=0)):
                # print('NW Step')
                temp[0] -= 1
                temp[1] -= 1
                edge = False
            else:
                # print(step,'Edge condition')
                edge = True

        cur = temp 
        # if ((cur[0] < 0) or (cur[0] > size) or (cur[1] < 0) or (cur[1] > size)):
        #     print('Next step',step, cur)
        # print('cur is now',cur)
        walk.append(cur)
        # print('walk',walk)
        # walk.append(temp)
        edge = True  
    return walk

imgWalk_dict = {}
def trajectory_to_frames(im_ix = 0, n_steps = CAMERA_RES[0] * CAMERA_RES[1], size = IM_SIZE, sensor_size = CAMERA_RES, record = True):
    img = np.array(DATASET[im_ix][0])
    walk = imgRandomWalk(n_steps = n_steps, size = sensor_size)
    
    if record: 
        imgWalk_dict[im_ix] = walk
    
    # frames = [img[:,step[0]:step[0]+frame_size,step[1]:step[1]+frame_size] for step in walk]
    # frames = [img[step[0]:step[0]+frame_size,step[1]:step[1]+frame_size,:] for step in walk]
    
    frames = [np.zeros(CAMERA_RES,dtype=int) for step in walk]
    for i in range(len(frames)):
        frames[i][walk[i][0]:walk[i][0]+IM_SIZE,walk[i][1]:walk[i][1]+IM_SIZE] = DATASET[im_ix][0]
    
    # for step in walk:
    #     img = img[:,ste[0]:step[0]]
    
    return frames




if __name__ == '__main__':
    # 900 frames at 30fps is 30 seconds - ?
    # drift speed ~40 arcmin/second - 
    # don't record all the steps 
    im_ixs = [i for i in range(len(DATASET))]
    # random.sample - without replacement (random.choices with replacement)
    ixs = sample(im_ixs,16)

    for i in ixs:
        moV_filename = f"Im_{i}"
        frames_smbatch = trajectory_to_frames(im_ix=i,n_steps=300,record=False)
        frames_smbatch = np.stack(frames_smbatch, axis=0)
        video_filepath = os.path.join(OUTDIR,moV_filename)
        # 40Hz for 0.025 duration conversion - no interstep event mushing
        write_video(video_filepath,frames_smbatch,fps=50)
        
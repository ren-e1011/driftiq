import os
import pickle
from copy import deepcopy
from random import shuffle, sample, choice

import numpy as np
from torchvision.io import write_video

from matplotlib import pyplot as plt

import frames2events

from envar import *

# in principle could process the image as a lawnmower
# n_steps OUT_DIM**2 for lawnmower is overkill - find some basis/justification for step count. works on full set. trial on small amount for testing
# choose some step count in which probabilistically all <24> *patches will be traversed moving pixelwise -- considering global attention is patch-by-patch

# n_steps = args.n_frames, size = CAMERA_RES
def imgRandomWalk(args, size = CAMERA_RES, im_size = 32):

    n_steps = args.n_frames

    size = (size,size) if isinstance(size, int) else size
    # NW (upper left) coordinate
    # first dimension height. second dimension width
    # MOD init should not roll over the S/E end
    init_tup = [choice(range(size[0]-im_size)),choice(range(size[1]-im_size))]
    # for now initialize them all uniformly so network does not learn placement:class "association"? but there are many of the same class and sparse representation should take care of that?
    # init_tup = [size[0]//2,size[1]//2]
    walk = []
    # cur = init_tup
    cur = deepcopy(init_tup)
    # print('init_tup',cur)
    # starting point is governed by saccade - collect following
    # walk.append(cur)

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

            # + im_size incorporates +1    
            elif ((step == 'NE') and (temp[0] - 1 >= 0) and (temp[1] + im_size < size[1])):
                # print('NE Step')
                temp[0] -= 1
                temp[1] += 1
                edge = False
                
            elif ((step == 'E') and (temp[1] + im_size < size[1])):
                # print('E Step')
                temp[1] += 1
                edge = False 
                
            elif ((step == 'SE') and (temp[0] + im_size < size[0]) and (temp[1] + im_size < size[1])):
                temp[0] += 1
                temp[1] += 1
                edge = False 
                
            elif ((step == 'S') and (temp[0] + im_size < size[0])):
                # print('S Step')
                temp[0] += 1
                edge = False
                
            elif ((step == 'SW') and (temp[0] + im_size < size[0]) and (temp[1] - 1 >= 0)):
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

        walk.append(cur)

        edge = True  

    return walk


def randTraj2frames(args, imix, im_size = 32, sensor_size = (260,346,3), record = False):

    n_steps = args.n_frames

    img = np.array(DATASET[imix][0])
    # print('img_size',img.size)

    walk = imgRandomWalk(args, size = sensor_size)
    # print(len(walk),'steps')
    
    # TODO fix or rm 
    if record: 
        imgWalk_dict = {}
        imgWalk_dict[imix] = walk
    
    frames = [np.zeros(sensor_size,dtype=int) for step in walk]
    # print(len(frames),'frames')
    # print(frames[0].shape, 'frame_shape')
    for i in range(len(frames)):
        frames[i][walk[i][0]:walk[i][0]+im_size,walk[i][1]:walk[i][1]+im_size] = img

    # presumes return format
    frames_smbatch = np.stack(frames, axis=0)

    return frames_smbatch


def pickleRandomWalk(args,imix, frames:np.array = None):

    if not frames:
        frames = randTraj2frames(args, imix)

    if args.save_walk_video:
        moV_filename = f"Im_{imix}.mov"
        write_video(os.path.join(args.video_path,moV_filename),frames,fps=args.frame_rate_hz)

    
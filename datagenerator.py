# generate dataset or 
import argparse
from subprocess import Popen

import os
from random import sample

from frames2events_subp import im2frames2events

from envar import * 

os.chdir(FILEPATH)


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
parser.add_argument("--n_im", type=int, default=None)
# Clean vs noisy for events generation - not relevant for this project
parser.add_argument("--condition",type=str,default="Clean")

args = parser.parse_args()

def generate_randomwalk_events(im_ixs):

    proc_ixs = [i for i in im_ixs if not os.path.isfile(args.events_path+f"/Im_{i}.avi")]
        # test 
    
    # mod to change 
    # im2frames2events returns the command
    procs = [Popen(im2frames2events(args,i)) for i in proc_ixs]

    for p in procs:
        p.wait()

def generate_infodriven_events(im_ixs):

    raise NotImplementedError

if __name__ == '__main__':
    # 900 frames at 30fps is 30 seconds - ?
    # drift speed ~40 arcmin/second - 
    # don't record all the steps 
    im_ixs = [i for i in range(len(CIFAR))]
    # random.sample - without replacement (random.choices with replacement)
    if args.n_im is not None:
        im_ixs = sample(im_ixs,args.n_im)

    if args.walk == 'random':
        generate_randomwalk_events(im_ixs)
    
    # create events if does not exist
    
        
    
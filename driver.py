# wrapper for RVT > train.py
from envar import *
import os
os.chdir(FILEPATH)


os.environ["HYDRA_FULL_ERROR"] = "1"

from dataset import DataSet
from torch.utils.data import DataLoader
import argparse


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


import hydra
from omegaconf import DictConfig, OmegaConf
from RVTClass.models.classification.yolox_extension.models.classifier import Classifar

# @hydra.main(config_path='./RVTClass/config', config_name='train', version_base='1.2')
def main():
    dataset = DataSet(args)
    dataloader = DataLoader(dataset, batch_size=1)

    for X,y in dataloader:
        break
    
    print(X.shape, y.shape)

    mdl = Classifar()
    return X,y

if __name__ == "__main__":
    main()

    
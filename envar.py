from torchvision import datasets, transforms
import numpy as np
import os

FILEPATH = '/home/renee/DIQ'
OUTDIR = './Data/RandomImWalk/300Frames50Hz/'
# swapped 346, 260 by inspection...shouldnt matter
CAMERA_RES = (260,346,3)
# default SENSOR.dtype is dtype('int64')
SENSOR = np.zeros(CAMERA_RES,dtype=int)

CIFAR = datasets.CIFAR100(os.path.join(FILEPATH,'Data/'),train=True,download=True, transform=None)
IM_SIZE = CIFAR[0][0].size[0]

# if self.train else downloaded_list = self.test_list - butt there is a test folder?
# TESTSET = DATASET = datasets.CIFAR100(os.path.join(FILEPATH,'Data/'),train=False,download=True, transform=None)

BATCH_SIZE = 4
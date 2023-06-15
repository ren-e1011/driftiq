from envar import *
import os
from torch.utils.data import DataLoader

os.chdir(FILEPATH)





data_loader = DataLoader(CIFAR,batch_size = BATCH_SIZE, shuffle=True)
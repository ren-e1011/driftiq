from copy import deepcopy
from random import choice

import numpy as np

from envar import *

class RandomWalk:
    def __init__(self,sensor_size= CAMERA_RES, im_size = IM_SIZE):

        size = (sensor_size,sensor_size) if isinstance(sensor_size, int) else sensor_size
        # same starting point for all instances, experiments - center of sensor
        # init_tup
        self.x_a = [size[0]//2 - im_size//2, size[1]//2 - im_size//2]
        self.sensor_size = sensor_size
        self.im_size = im_size

        # save trajectory
        self.walk = []
        # initialized by imix
        #self.v2ee = eventemulator
        #self.img = np.array(CIFAR[imix][0])

        # FIRST HITS
        #i = -1
        
        #while  i not in self.v2ee[imix]['step'].keys():
        self.stepset = ['N','E','S','W', 'NE', 'NW', 'SE', 'SW']
    

    def coord_move(self,vec:str = None, x_a:list = [], stepset:list = None):
            
        x_a = self.x_a if x_a == [] else x_a

        #edge = False
        
        stepset = deepcopy(self.stepset) if stepset is None else stepset

        # random step
        vec = choice(stepset) if vec is None else vec

        # random step
        # if vec is None:
        #     stepset = deepcopy(self.stepset)
        #     vec = choice(stepset)

        if vec == 'N' and (x_a[0] - 1 >= 0):
            x_a[0] -= 1
            
        elif vec == 'NE' and (x_a[0] - 1 >= 0) and (x_a[1] + self.im_size < self.sensor_size[1]):
            x_a[0] -= 1
            x_a[1] += 1

        elif vec == 'E' and (x_a[1] + self.im_size < self.sensor_size[1]):
            x_a[1] += 1

        elif vec == 'SE' and (x_a[0] + self.im_size < self.sensor_size[0]) and (x_a[1] + self.im_size < self.sensor_size[1]):
            x_a[0] += 1
            x_a[1] += 1

        elif vec == 'S' and (x_a[0] + self.im_size < self.sensor_size[0]):
            x_a[0] += 1

        elif vec == 'SW' and (x_a[0] + self.im_size < self.sensor_size[0]) and (x_a[1] - 1 >= 0):
            x_a[0] += 1
            x_a[1] -= 1

        elif vec == 'W' and (x_a[1] - 1 >= 0):
            x_a[1] -= 1 

        elif vec == 'NW' and (x_a[0] - 1 >= 0) and (x_a[1] - 1 >=0):
            x_a[0] -= 1
            x_a[1] -= 1
        # TODO test - as a while edge == true
        else: 
            #edge = True
            # if edge, pick a random step 
            # would work for removing just one edge and also if the img is eg in a corner there are 3 
            stepset = list(filter(lambda x: x != vec, stepset))
            # vec is not None so will not instantiate new stepset
            self.coord_move(vec=choice(stepset),x_a=x_a,stepset=stepset)
            
            # coord_move(<- if edge, pick a new )

        return x_a
    
if __name__ == "__main__":
    rw = RandomWalk()
    xa = [191,319]
    xa1 = rw.coord_move(x_a = xa)
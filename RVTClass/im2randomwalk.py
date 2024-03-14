from copy import deepcopy
from random import choice
from utils import _coord_move



from envar import IM_SIZE, CAMERA_RES
import numpy as np

class RandomWalk:
    def __init__(self,sensor_size= CAMERA_RES, im_size = IM_SIZE, start_pos:list = []):

        size = (sensor_size,sensor_size) if isinstance(sensor_size, int) else sensor_size
        
        # mod self.sensor_size = size 
        self.sensor_size = size
        self.im_size = im_size

        # same starting point for all instances, experiments - center of sensor
        # init_tup
        self.start_pos = [size[0]//2 - im_size//2, size[1]//2 - im_size//2] if not start_pos else start_pos
        # save trajectory
        # starting with the start_pos presumes reinstantiating walk without memory 
        # will need to refactor if refractory_period >> 0 st the system has memory
        self.walk = [self.start_pos]
        # self.walk.append(x_a)
        
        #while  i not in self.v2ee[imix]['step'].keys():
        self.stepset = ['N','E','S','W', 'NE', 'NW', 'SE', 'SW']
    
    
    
    # x_a is an input starting position - for continuing a walk 
    def coord_move(self,vec:str = None, x_a:list = [], stepset:list = None):
            
        x_a = self.walk[-1].copy() if not x_a else x_a
        
        stepset = deepcopy(self.stepset) if stepset is None else stepset

        # random step
        vec = choice(stepset) if vec is None else vec

        x_a_next, _ = _coord_move(vec, x_a, stepset = stepset, sensor_size = self.sensor_size)
        # self.walk.append(x_a_next)
        
        return x_a_next
    
    def update(self, x_a_next, nspikes):
        self.walk.append(x_a_next)
    
if __name__ == "__main__":
    rw = RandomWalk()
    xa = [55,2]
    xa1 = rw.coord_move(x_a = xa)
    for step in range(4):
        xa1 = rw.coord_move()
    print(xa1)
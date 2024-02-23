from copy import deepcopy
from random import choice



from envar import *
import numpy as np

class RandomWalk:
    def __init__(self,sensor_size= CAMERA_RES, im_size = IM_SIZE, start_pos:list = []):

        size = (sensor_size,sensor_size) if isinstance(sensor_size, int) else sensor_size
        
        
        self.sensor_size = sensor_size
        self.im_size = im_size

        # same starting point for all instances, experiments - center of sensor
        # init_tup
        x_a = [size[0]//2 - im_size//2, size[1]//2 - im_size//2] if not start_pos else start_pos
        # save trajectory
        # starting with the start_pos presumes reinstantiating walk without memory 
        # will need to refactor if refractory_period >> 0 st the system has memory
        self.walk = [x_a]
        # self.walk.append(x_a)
        
        #while  i not in self.v2ee[imix]['step'].keys():
        self.stepset = ['N','E','S','W', 'NE', 'NW', 'SE', 'SW']

    def _coord_move(self,vec: str, x_a: list, stepset: list):
        if vec == 'N' and (x_a[1] - 1 >= 0):
            x_a[1] -= 1
            # 
            
        elif vec == 'NE' and (x_a[1] - 1 >= 0) and (x_a[0] + self.im_size < self.sensor_size[0]):
            x_a[1] -= 1
            x_a[0] += 1

        elif vec == 'E' and (x_a[0] + self.im_size < self.sensor_size[0]):
            x_a[0] += 1

        elif vec == 'SE' and (x_a[1] + self.im_size < self.sensor_size[1]) and (x_a[0] + self.im_size < self.sensor_size[0]):
            x_a[1] += 1
            x_a[0] += 1

        elif vec == 'S' and (x_a[1] + self.im_size < self.sensor_size[1]):
            x_a[1] += 1

        elif vec == 'SW' and (x_a[1] + self.im_size < self.sensor_size[1]) and (x_a[0] - 1 >= 0):
            x_a[1] += 1
            x_a[0] -= 1

        elif vec == 'W' and (x_a[0] - 1 >= 0):
            x_a[0] -= 1 

        elif vec == 'NW' and (x_a[1] - 1 >= 0) and (x_a[0] - 1 >=0):
            x_a[1] -= 1
            x_a[0] -= 1
            # stand still 
        elif vec == 'X':
            x_a = x_a 
        else: 
            # if edge, pick a random step 
            # would work for removing just one edge and also if the img is eg in a corner there are 3 
            stepset = list(filter(lambda x: x != vec, stepset))
            # vec is not None so will not instantiate new stepset
            self._coord_move(vec=choice(stepset),x_a=x_a,stepset=stepset)
            # print(f"out of frame, step {step}")
            # coord_move(<- if edge, pick a new )
        
        return x_a
    
    # x_a is an input starting position - for continuing a walk 
    def coord_move(self,vec:str = None, x_a:list = [], stepset:list = None):
            
        x_a = self.walk[-1].copy() if not x_a else x_a
        
        stepset = deepcopy(self.stepset) if stepset is None else stepset

        # random step
        vec = choice(stepset) if vec is None else vec

        x_a_next = self._coord_move(vec, x_a, stepset)
        self.walk.append(x_a_next)
        
        return x_a_next
    
if __name__ == "__main__":
    rw = RandomWalk()
    xa = [55,2]
    xa1 = rw.coord_move(x_a = xa)
    for step in range(4):
        xa1 = rw.coord_move()
    print(xa1)
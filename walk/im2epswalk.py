from copy import deepcopy
from random import choice
import numpy as np
from utils.utils import _coord_move, _traj_to_dir
from configs.envar import IM_SIZE, CAMERA_RES
from numpy.random import lognormal
from math import log, exp

# snippet from https://www.geeksforgeeks.org/epsilon-greedy-algorithm-in-reinforcement-learning/

class Action: 
    def __init__(self, action: str, sigma_): 
        self.i = action
        self.mu = 0
        self.n = 0

    # Update the action-value estimate 
    def update(self, hits): 
        self.n += 1
        # self.mean = (1 - 1.0 / self.N)*self.mean + 1.0 / self.N * hits 
        self.mu = ((self.n - 1) / float(self.n)) * self.mu + (1.0 / float(self.n)) * hits
        
            
class EPSWalk:
    def __init__(self,sensor_size= CAMERA_RES, im_size = IM_SIZE, maximize = True, start_pos:list = [], eps=0.02): # todo rm w

        size = (sensor_size,sensor_size) if isinstance(sensor_size, int) else sensor_size
        
        self.sensor_size = size
        self.im_size = im_size

        # same starting point for all instances, experiments - center of sensor
        self.start_pos = [size[0]//2 - im_size[0]//2, size[1]//2 - im_size[1]//2] if not start_pos else start_pos
        # save trajectory
        # starting with the start_pos presumes reinstantiating walk without memory 
        # will need to refactor if refractory_period >> 0 st the system has memory
        self.walk = [self.start_pos]
        
        self.stepset = ['N','E','S','W', 'NE', 'NW', 'SE', 'SW']

        self.actions = [Action(a) for a in self.stepset] 
        self.maximize = maximize
        self.eps = eps


        # x_a is an input starting position - for continuing a walk 
    def coord_move(self,vec:str = None, x_a:list = [], stepset:list = None):
            
        x_a = self.walk[-1].copy() if not x_a else x_a
        
        stepset = deepcopy(self.stepset) if stepset is None else stepset

        # presumes warmup 
        if vec is None:
            assert np.count_nonzero([a.n for a in self.actions]) == len(self.actions)
            # eps
            p = np.random.random() 
            if p < self.eps: 
                j = np.random.choice(self.actions)
                vec = j.i 
            else: 
                j = np.argmax([a.mu for a in self.actions]) 
                vec = self.actions[j].i

        x_a_next, _ = _coord_move(vec, x_a, stepset = stepset, sensor_size = self.sensor_size)
        
        return x_a_next
    
    def update(self, x_a_next, nspikes):

        if not self.maximize:
            nspikes = -nspikes

        vec = _traj_to_dir(imtraj=[self.walk[-1].copy(),x_a_next.copy()])[0]
        act = np.where([a.i == vec for a in self.actions])[0].item()
        self.actions[act].update(nspikes)

        self.walk.append(x_a_next)

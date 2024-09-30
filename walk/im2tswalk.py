from copy import deepcopy
from random import choice
import numpy as np
from utils.utils import _coord_move, _traj_to_dir
# from configs.envar import IM_SIZE, CAMERA_RES
from numpy.random import normal
from math import log, exp

# snippet from https://www.geeksforgeeks.org/epsilon-greedy-algorithm-in-reinforcement-learning/

class Action: 
    def __init__(self, action: str, sigma_: int = 50, mu_: int = 500): 
        self.i = action
        self.mu = mu_ 
        self.n = 0
        self.sigma_ = sigma_ 
        self.sigma = sigma_ # variance on the order of 100     
    # # Choose a random action 
    # def choose(self):  
    #     return np.random.randn() + self.m 
    
    # Update the action-value estimate 
    def update(self, nhits): 
        self.n += 1
        # self.mean = (1 - 1.0 / self.N)*self.mean + 1.0 / self.N * hits 
        # self.mu = ((self.n - 1) / float(self.n)) * self.mu + (1.0 / float(self.n)) * hits
        if nhits > 0:
            self.mu = (self.sigma*self.mu + self.sigma_*nhits) / (self.sigma + self.sigma_)
            self.sigma = (self.sigma*self.sigma_)/(self.sigma+self.sigma_)

class TSWalk:
    def __init__(self,sensor_size= 96, im_size = 32, maximize = True, start_pos:list = [], cdp = 50, mu=500): # todo rm w

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

        self.actions = [Action(a,cdp,mu) for a in self.stepset] 
        self.maximize = maximize

        # self.mu_std = {a:[] for a in self.stepset}
        # self.mu_std = [] # FOR TESTING


        # x_a is an input starting position - for continuing a walk 
    def coord_move(self,vec:str = None, x_a:list = [], stepset:list = None):
            
        x_a = self.walk[-1].copy() if not x_a else x_a
        
        stepset = deepcopy(self.stepset) if stepset is None else stepset

        # presumes warmup 
        if vec is None:
            # assert np.count_nonzero([a.n for a in self.actions]) == len(self.actions) mod warmup false
            # Thompson Sampling 
            j = np.argmax([normal(a.mu,a.sigma) for a in self.actions])
            vec = self.actions[j].i    

        x_a_next, _ = _coord_move(vec, x_a, stepset = stepset, sensor_size = self.sensor_size)
        
        return x_a_next
    
    def update(self, x_a_next, nspikes):

        if not self.maximize:
            nspikes = -nspikes

        vec = _traj_to_dir(imtraj=[self.walk[-1].copy(),x_a_next.copy()])[0]
        act = np.where([a.i == vec for a in self.actions])[0].item()
        self.actions[act].update(nspikes)

        # self.mu_std.append((self.actions[act].mu,self.actions[act].sigma)) # FOR TESTING

        self.walk.append(x_a_next)

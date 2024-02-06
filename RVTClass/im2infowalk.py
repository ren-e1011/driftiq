import os
import pickle
from copy import deepcopy
from random import choice
from decimal import Decimal

import numpy as np
from scipy import special 

from envar import *
#from im2randomwalk import imgRandomStep

#from frames2events_emulator import StatefulEmulator

from math import factorial, exp, log2, prod


class InfoWalk(object):
    # note that sensor size is in dimensions y,x for v2e input while prior was recorded in x,y
    def __init__(self,sensor_size= CAMERA_RES, im_size = IM_SIZE, p_prior: np.array = None, mean_spikes: np.array = None):

        size = (sensor_size,sensor_size) if isinstance(sensor_size, int) else sensor_size
        # same starting point for all instances, experiments - center of sensor
        # init agent position x_a
        self.x_a = [size[0]//2 - im_size//2, size[1]//2 - im_size//2]

        self.sensor_size = size
        self.im_size = im_size

        # save trajectory
        self.walk = []
        # initialized by imix
        #self.v2ee = eventemulator
        #self.img = np.array(CIFAR[imix][0])

        # FIRST HITS
        if mean_spikes is None:
            mean_spikes = np.ones((self.sensor_size[0],self.sensor_size[1]))
            min = 1 
        else:
            min = int(mean_spikes[mean_spikes > 1].min())
        
        self.mean_spikes = mean_spikes
    
        max = int(mean_spikes.max() + 1)
        #i = -1
        # 1 hit likelihood (==0) in steps of min 
        # min 62
        # max 425
        self.hit_list = range(1,max,min)
        
        #while  i not in self.v2ee[imix]['step'].keys():
        self.stepset = ['N','E','S','W', 'NE', 'NW', 'SE', 'SW']
    
        # prior on source probability distribution 
        # uniform prior...or random walk prior 
        if p_prior is None: 
            p_prior = np.ones((self.sensor_size[0],self.sensor_size[1]))
            # interchangeable with self.p_prior = self.p_prior / prod((sensor_size[0],sensor_size[1]))
            p_prior = p_prior / p_prior.sum()

        self.p_prior = p_prior 
        # converts [[1,2],[1,3],[2,3]] to ((1, 1, 2), (2, 3, 3)) to slice an array 
        #walk_indices = tuple(zip(*self.walk))

        # uniform prior at t_0
        #self.p_prior[walk_indices] = 0.0
        # probability of source at a given location 
        
        

    # vec defines action a 
    def coord_move(self,vec:str = None, x_a:list = [], stepset:list = None, edge_correct = True):
        
        # updates self.x_a or passed in coordinates
        x_a = self.x_a if x_a == [] else x_a
        
        stepset = deepcopy(self.stepset) if stepset is None else stepset

        # random step
        vec = choice(stepset) if vec is None else vec

        if vec == 'N'and (x_a[0] - 1 >= 0):
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
        else: # edge
            if not edge_correct:
                return
            # if edge, pick a random step :: test!
            # shouldnt get stuck in a loop...
            # would work for removing just one edge and also if the img is eg in a corner there are 3 
            stepset = list(filter(lambda x: x != vec, stepset))
            self.coord_move(vec=choice(stepset),x_a=x_a,stepset=stepset)
            
            # coord_move(<- if edge, pick a new )

        return x_a

    # At each step (source not found)
    # h - number of hits received
    # x_a - location of agent
    # x_set - x' alternative locations of 
    def bayes_update(self,h: int, x_a: tuple, update = False):
        # st update is inplace 
        prior = self.p_prior if update == True else deepcopy(self.p_prior)

        # first update the prior -> posterior (prior for next step) 
        # "hit not found ie continue" - decision to be modified in the future
        
        # prior[x_a] = 0.0
        # to calculate entropy
        prior[x_a] = 1.0
        # renormalization
        prior = prior / prior.sum()

        # does not work
        # mean_spikes = np.asarray(self.mean_spikes(), dtype=Decimal)

        # elementwise
        # poisson = (self.mean_spikes ** h * exp(-self.mean_spikes)) / factorial(h)

        # requires decimal for large h which cannot be done elementwise in np array 
        # poisson = ((mean_spikes ** h) * np.exp(-mean_spikes)) / special.factorial(h)

        bayes_posterior = [[None for y in range(prior.shape[1])] for x in range(prior.shape[0])]
        running_sum = Decimal(0)
        for x in range(prior.shape[0]):
            for y in range(prior.shape[1]):
                _prior = Decimal(prior[(x,y)])
                # Don’t call .item() anywhere in your code. Use .detach() instead to remove the connected graph calls. Lightning takes a great deal of care to be optimized for this. (pl docs)
                #  https://lightning.ai/docs/pytorch/stable/advanced/speed.html
                _poisson = Decimal(self.mean_spikes[(x,y)].item()) 
                _poisson = _poisson ** h
                _poisson *= Decimal(-self.mean_spikes[(x,y)].item()).exp()
                _poisson /= Decimal(factorial(h))

                bayes_posterior[x][y] = (_poisson * _prior) 
                running_sum += bayes_posterior[x][y]

        for x in range(prior.shape[0]):
            for y in range(prior.shape[1]):
                bayes_posterior[x][y] = (bayes_posterior[x][y] / running_sum)

        bayes_posterior = np.array(bayes_posterior)
        # does a 2d array slice like that
        # bayes_posterior = (poisson * prior) / (poisson * prior).sum()

        # posterior 
        # refractor to integrate into HSA 
        return bayes_posterior
        # when a step is taken
        #self.p_prior = bayes_posterior

    # entropy of belief state s = [x_a, p(x)] - agent and prior/posterior distribution of source
    # independent of agent position x_a
    # Shannon entropy H(s) as per (3.1) in Loisy & Eloy 2022
    def H_s(self, prior = None):
        prior = self.p_prior if prior is None else prior
        
        # measures certainty of source location/orderliness of space
        # np.log2 is elementwise. prior * np.log2(prior) is elementwise. np.sum with no axis given sums all elements in array 
        return (prior * np.log2(prior)).sum()
    
    # expected entropy of successor belief states s' :: expected entropy upon taking a in belief state s
    # H(s|a) given by (3.2) in Loisy & Eloy and section 4 in supplementary materials
    # stepdir is one of self.stepset
    def H_s_a(self,a):
        
        #for a in self.stepset:
        # x_a = deepcopy(self.x_a)
        x_a = list(self.walk[-1].copy())
        # could be None 
        x_t1 = self.coord_move(vec = a,x_a = x_a, edge_correct= False)

        if x_t1 is None:
            return 
        
        
        # if want to explicitly prevent a rerun if traversed before
        # if self.prior[x_t1] == 0:
        #     return None

        # for indexing 
        x_t1 = tuple(x_t1)

        prior_bar = deepcopy(self.p_prior)
        # probability agent does not find source 
        p_bar = 1 - prior_bar[x_t1]
        # -> -inf H_s at this location 
        # replace with 1 else how do they calculate the entropy 
        # prior_bar[x_t1] = 0.0
        prior_bar[x_t1] = 1.0

        # prior bar
        # renormalization
        prior_bar = prior_bar / prior_bar.sum()

        hsa_sum = Decimal(0.0)
        for h in self.hit_list:
            # Pr(h | 𝜇)
            # 𝜇 is the mean number of hits. It is a function of the Euclidean distance 𝑑=‖𝑥𝑠−𝑥𝑎‖2
            mean_spikes = int(self.mean_spikes[x_t1])
            poisson_h_xt1 = (Decimal(mean_spikes ** h) * Decimal(exp(-mean_spikes))) / Decimal(factorial(h)) 
            H_h = Decimal(self.H_s(prior_bar))
            hsa_sum += (poisson_h_xt1 * H_h)

        #p_bar = 1 - prior[x_t1]

        return x_t1, hsa_sum * Decimal(p_bar) 



    # information gain with taking action a in belief state s - not used but theoretical 
    # def G_s_a(self,a):

    #     return self.H_s() - self.H_s_a(a)

    # choose the step which maximizes information gain by minimizing expected entropy H(s|a)
    # ^ information gain G = H(s) - H(s|a)
    def next_step(self, h: int, x_a = []):

        # xa_next
        min_ent = ([],np.inf)
        for vec in self.stepset:
            hsa = self.H_s_a(vec)
            # hit an edge
            # ...or been their before (uncomment ^). leave out in case agent is gridlocked
            if hsa is None:
                continue
            else:
                xa_next, entropy_a = hsa
                if entropy_a < min_ent[1]:
                    min_ent = (list(xa_next), entropy_a)

        # self.x_a = xa_next
        self.walk.append(min_ent[0])
        
        self.bayes_update(h, x_a = xa_next, update = True)

        #return min_ent[0]
        # convert back to list form for coord_next 
        
        # to save entropy 
        return min_ent


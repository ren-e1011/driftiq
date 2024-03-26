from copy import deepcopy
from random import choice
import numpy as np
from utils.utils import _coord_move
from configs.envar import IM_SIZE, CAMERA_RES, EPSILON
from scipy.stats import poisson as Poisson_
from warnings import warn 
from scipy.special import kn

class InfoWalk:
    # note that sensor size is in dimensions y,x for v2e input while prior was recorded in x,y
    def __init__(self,sensor_size= CAMERA_RES, im_size: int = IM_SIZE, start_pos:list = [], p_prior: np.array = None, mean_spikes: np.array = None):

        self.sensor_size = (sensor_size,sensor_size) if isinstance(sensor_size, int) else sensor_size
        self.im_size = (im_size, im_size) if isinstance(im_size, int) else im_size

        # (65,65)
        self.Ngrid =(self.sensor_size[0] - self.im_size[0] + 1 , self.sensor_size[1] - self.im_size[1] + 1) # int((Ngrid // 2) * 2 + 1) where Ngrid == CAMERA_RES[0] == CAMERA_RES[1] (96) to ensure odd value
        self.N = max(self.Ngrid) # 65

        # init agent position x_a - center of sensor - unless restarting a walk at start_pos
        # self.start_pos = [self.sensor_size[0]//2 - self.im_size[0]//2, self.sensor_size[1]//2 - self.im_size[1]//2] if not start_pos else start_pos
        self.start_pos = [self.Ngrid[0]//2, self.Ngrid[1]//2] if not start_pos else start_pos
        # save trajectory as list of NW coordinates
        self.walk = [self.start_pos]
        # possible directions 
        self.stepset = ['N','E','S','W', 'NE', 'NW', 'SE', 'SW']
        # prior probability of direction of "source" at a given location  
        # uniform prior...or random walk prior 
        if p_prior is None: 
            # p_prior = np.ones([self.sensor_size[0] - self.im_size[0] + 1] + [self.sensor_size[1] - self.im_size[1] + 1])
            p_prior = np.ones([self.N,self.N]) #TODO mod for nonsquare condition 
            p_prior = p_prior / p_prior.sum()
        self.entropy = [-(p_prior * np.log2(p_prior)).sum()]
        self.p_prior = p_prior 

        # from otto evaluate.parameters._defaults.py
        self.lambda_over_dx = 2.0 # for shunting expected spikes by distance 
        self.Ndim = len(self.p_prior.shape) #2
        # why does otto initiate as -np.ones
        # 65
        # self.N = self.sensor_size[0] - self.im_size[0] + 1
        # self.hit_map = np.ones([self.N] * 2, dtype=int)
        self.norm = 'Euclidean'
        # converts [[1,2],[1,3],[2,3]] to ((1, 1, 2), (2, 3, 3)) to slice an array 
        #walk_indices = tuple(zip(*self.walk))
        self.leaving_p = 0.0

# otto sourcetracking._distance()
        # norm = Euclidean
        # returns relative euclidean distance from the origin to each point on the search space 
    def _distance_mesh(self, origin: tuple = None, Ndim: int = None, N: int = None):
        Ndim = self.Ndim if not Ndim else Ndim # 2
        N = self.N if not N else N # 65
        origin = tuple(self.start_pos) if origin is None else origin # (32,32)

        # (65,65)
        coord = np.mgrid[tuple([range(N)] * Ndim)] 
        # relative -,+ distance to origin
        for i in range(Ndim):
            coord[i] -= origin[i]
        d = np.zeros([N] * Ndim)

        for i in range(Ndim):
            d += (coord[i]) ** 2
        d = np.sqrt(d)

        #(65,65)
        return d

    # makes no assumptions on origin story of hits - after all, there is no source. Instead just 
    def _mean_number_of_hits(self, distances, mu_spikes: int = None):
        if mu_spikes is None:
            assert hasattr(self,'mu')
            mu_spikes = self.mu
        distances = np.array(distances)
        distances[distances == 0] = 1.0 # from 0 to 1 to avoid inf value in next step. will set to 0 in _init_params
        
        # expected spikes decreases with distance
        mu = kn(0, distances / self.lambda_over_dx) / kn(0, 1) # TODO is this the appropriate shunting computation for a sourceless search 
        # mu *= self.mu
        mu *= mu_spikes
        return mu
    ## modified version of otto sourcetracking.py _Poisson(), _Poisson_unbounded
    def _Poisson(self, mu, h):
        assert hasattr(self, 'hmax')
        # 
        # if h >= self.hmax:
        #     warn(f"{h} hits h larger than hmax, {self.hmax}. Setting h to hmax")
        #     h = self.hmax - 1
        
        if h < self.hmax - 1:
            p = Poisson_(mu).pmf(h)
        # elif passed in h >= self.hmax - 1, return the rest of the probability density. effectively thresholding h at hmax
            
        # means of thresholding spikes at self.hmax 
        else: 
            p = 1 - Poisson_(mu).cdf(self.hmax - 2)
        # why cant i just mod hmax 
            
        return p 

    def _init_params(self,mean_spikes,maxspikes, std_spikes):
        self.mu = mean_spikes 

        # MOD
        self.hmax = int(mean_spikes + std_spikes)
        # alternatives to hmax 
        # self.hmax = int(mean_spikes + np.sqrt(mean_spikes)) 
        # self.hmax = max(self.hmax, maxspikes + 1)

        # # if maxspikes > self.hmax:
        # #     # instead of raising hmax, ceiling of hmax 
        # #     warn(f"Max spikes, {maxspikes}, is larger than mu + sqrt(mu), {self.hmax}.")
        # #     self.hmax = maxspikes
        # d = self._distance_mesh()
        # mu = self._mean_number_of_hits(d)
        # mu[tuple(self.start_pos)] = 0.0
        # ## snippet from otto sourcetracking.py _compute_p_Poisson() lines 363-393
        # # probability of receiving hits Pr(h|xa,x')
        # # (len(hit_list), 65, 65) - for each NW coordinate 
        # # self.p_Poisson = np.zeros([self.hmax] + [self.sensor_size[0] - self.im_size[0] + 1] + [self.sensor_size[1] - self.im_size[1] + 1])
        
        # self.p_Poisson = np.zeros([self.hmax]+[self.sensor_size[0]]+[self.sensor_size[1]])

        # logic from _compute_p_Poisson to be used for computing the posterior squared on each element of p_prior 
        size = 2 * self.N - 1 # 129 
        origin = [self.N] * self.Ndim # (65,65)
        origin = tuple(origin)
        distance_mx = self._distance_mesh(origin=origin,N=size)
        mu = self._mean_number_of_hits(distance_mx, self.mu)
        mu[origin] = self.leaving_p # start position has zero probability TODO mod leaving probability 
        self.p_Poisson = np.zeros([self.hmax] + [size] * self.Ndim) # (hmax, 129, 129)

        # self.p_Poisson = np.zeros([self.hmax])
        # to test that the hmax threshold is not too high
        # if not square 
        # for square space == np.zeros([self.sensor_size[0] - self.im_size[0] + 1] * 2)
        # (65,65)
        # sum_proba = np.zeros([self.sensor_size[0] - self.im_size[0] + 1] + [self.sensor_size[1] - self.im_size[1] + 1])
        # sum_proba = np.zeros([self.sensor_size[0]] + [self.sensor_size[1]])
        sum_proba = np.zeros([size] * self.Ndim) # (129,129)
        # sum_proba = 0.
        # range(1,hmax + 1) to include hmax and minimize hits at 1 - became unnecessarily? complicated
        self.hit_range = range(self.hmax)
        for h in self.hit_range:
            # (hmax, 65, 65)
            self.p_Poisson[h] = self._Poisson(mu, h)

            sum_proba += self.p_Poisson[h]
            # if h is less than hmax - 1 and reached a probability of close to 1, reduce hmax 
            if h < self.hmax - 1:
                sum_is_one = np.all(abs(sum_proba - 1) < EPSILON)
                if sum_is_one:
                    warn(f"hmax at {self.hmax} is too large, reducing hmax to = {h} or lower - values higher than {h} have 0 probability")
                    self.hmax = h

        # by definition p_Poisson at> N (origin) = 0 
        self.p_Poisson[:, origin] = 0.0 # not self.leaving_p 
         

        # for all mu,h, pmf should == 1 
        # if not np.all(sum_proba == 1.0):
                    # TODO mod check 
        
        # MOD
        # if not np.isclose(sum(self.p_Poisson[:self.hmax]), 1.0): 
        #     warn(f"_compute_p_Poisson: sum proba is {sum_proba}, not 1")
            # sum_proba += self.p_Poisson[h]

        ## otto sourcetracking ln 400+
            # for all h set origin p to 0 ..redundant
         # by definition: p_Poisson(origin) = 0
        
        

    # At each step (source not found)
    # h - number of hits received
    # x_a - location of agent
    # x_set - x' alternative locations of 
    # mod with otto sourcetracking.py _update_p_source
    def bayes_update(self,nhits: int, x_a: tuple):
        # TODO get a sense of how many hits meet or exceed self.hmax 
        nhits = min(nhits, self.hmax-1)

        self.p_prior[x_a] = self.leaving_p # epsilon? 

        post_index = np.array([self.N - 1] * self.Ndim) - x_a  # (64,64) - (x_a[0],x_a[1])
        # p_evidence = self.p_Poisson[nhits][x_a[0]:x_a[0]+self.p_prior.shape[0],x_a[1]:x_a[1]+self.p_prior.shape[1]]
        p_evidence = self.p_Poisson[nhits,post_index[0]:post_index[0]+self.N,post_index[1]:post_index[1]+self.N] # (hmax,65,65)

        # self.p_prior *= self.p_Poisson[nhits]
        # for each h 
        self.p_prior *= p_evidence

        self.p_prior[self.p_prior <= EPSILON] = 0.0
        self.p_prior /= max(self.p_prior.sum(),EPSILON)

    
        # self.p_prior[x_a] = 1.0
        # why the second condition
        # to validate, probe entropy reduction 
        self.entropy.append(self.H_s())

        return self.p_prior 

    # entropy of belief state s = [x_a, p(x)] - agent and prior/posterior distribution of source
    # independent of agent position x_a
    # Shannon entropy H(s) as per (3.1) in Loisy & Eloy 2022
    def H_s(self, prior = None):
        prior = self.p_prior if prior is None else prior 
        # do not compute entropy for 0-probability spaces
        prior = prior[prior > EPSILON]
        # measures certainty of source location/orderliness of space
        # np.log2 is elementwise. prior * np.log2(prior) is elementwise. np.sum with no axis given sums all elements in array 
        # L & E 3.2. also see otto 456 np.sum(prior * -log)
        return -(prior * np.log2(prior)).sum()
    
    # expected entropy of successor belief states s' :: expected entropy upon taking action a from belief state s
    # H(s|a) given by (3.2) in Loisy & Eloy and section 4 in supplementary materials
    # stepdir is one of self.stepset
    def H_s_a(self,a, stepset):
  
        # x_a = deepcopy(self.x_a) - overkill. taken care of in _coord_move 
        x_a = deepcopy(list(self.walk[-1]))
        # will modify x_a but not self.walk[-1]
        x_t1, vec_select = _coord_move(vec = a, x_a = x_a, stepset = stepset, sensor_size = self.sensor_size)

        # if a random step is returned due to hitting the edge of the search space
        if vec_select != a:
            return 

        # for indexing 
        x_t1 = tuple(x_t1)

        # prior of space in the world where agent moves to a' 
        prior_bar = deepcopy(self.p_prior)
        # probability agent does not find source ie takes another step 
        p_bar = 1 - prior_bar[x_t1]
        # -> -inf H_s at this location 
        # replace with 1 else how to calculate the entropy 
        # prior_bar[x_t1] = 1.0
        # prior_bar[x_t1] = EPSILON
        prior_bar[x_t1] = self.leaving_p
        # renormalization
        prior_bar /= max(prior_bar.sum(),EPSILON) ## problematic
        
        # compute hit probability 
        # == np.zeros(self.hmax)

        # self.p_Poisson was initiated with self.hmax. self.hmax could have been degraded 
        hsa_mx = np.zeros(len(self.p_Poisson)) # hmax
        p_likelihood = self.p_Poisson[:,x_t1[0]:x_t1[0]+self.N,x_t1[1]:x_t1[1]+self.N]
        # hsa_mx = np.zeros(self.hmax)
        # hsa_sum = 0.0
        # hit_range is initiated with hmax but could possibly skip values 
        for h in self.hit_range:            
            # Pr(h | 𝜇) 
            # 𝜇 is the mean number of hits. It is a function of the Euclidean distance 𝑑=‖𝑥𝑠−𝑥𝑎‖2
            # mean_spikes = int(self.mean_spikes[x_t1])
            # factorial - might need decimal
            # poisson_h_xt1 = ((self.mu ** h) *(exp(-self.mu))) / (factorial(h))
            # H_h = self.H_s(prior_bar)
            # should never be neg? 
            # hsa_mx[h] = np.maximum(0, np.sum(prior_bar * self.p_Poisson[h]))
            hsa_mx[h] = np.maximum(0, np.sum(prior_bar * p_likelihood[h]))
            # hsa_sum += (poisson_h_xt1 * H_h)

        hsa_sum = hsa_mx.sum()

        hsa = hsa_sum * self.H_s(prior_bar)
        
        # not in otto but in l & e sm addt S25 ????
        hsa *= p_bar
        #p_bar = 1 - prior[x_t1]

        return x_t1, hsa

    # information gain with taking action a in belief state s - nself.muot used but theoretical 
    # def G_s_a(self,a):

    #     return self.H_s() - self.H_s_a(a)

    # choose the step which maximizes information gain by minimizing expected entropy H(s|a)
    # ^ information gain G = H(s) - H(s|a)
    # def random_start(self,x_a: list = [], vec: str = None):
    #     if len(self.mean_spikes_stepset) > 0:
    #         vec = choice(self.mean_spikes_stepset) if not vec else vec 
    #         x_a_next, _ = _coord_move(vec = vec, x_a = x_a, stepset = self.mean_spikes_stepset, sensor_size = self.sensor_size)
    #     else:

    #         return x_a_next 



    def coord_move(self, vec: list = [], x_a: list = []):
        # overkill. taken care of in _coord_move 
        x_a = deepcopy(self.walk[-1]) if not x_a else x_a
        stepset = deepcopy(self.stepset)

        if vec and vec in stepset:
            x_a_next = _coord_move(vec = vec, x_a = x_a, stepset = stepset, sensor_size= self.sensor_size)
            return x_a_next

        # coords, entropy
        min_ent = ([],np.inf)

        for vec in stepset:
            hsa = self.H_s_a(vec, stepset)
            # hit an edge
            # ...or been their before (uncomment ^). leave out in case agent is gridlocked
            if hsa is None:
                continue
            else:
                x_a_next, entropy_a = hsa
                if entropy_a < min_ent[1]:
                    # convert back to list form for coord_next 
                    min_ent = (list(x_a_next), entropy_a)
        return min_ent[0]
    

    def update(self, x_a_next, nspikes = 1):
        self.walk.append(x_a_next)
        
        # TODO mildly janky 
        # if not initiating, update prior 
        if hasattr(self,'mu'):
            self.bayes_update(nhits = nspikes, x_a = x_a_next)

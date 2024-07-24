from pathlib import Path 
import os, sys
sys.path.append(str(Path.cwd().parent)) # for pythonpath 
sys.path.append(str(Path.cwd()))

from configs.envar import *
import numpy as np
from Data.frames2events_emulator import StatefulEmulator
from walk.im2epswalk import EPSWalk
from walk.im2ucbwalk import UCBWalk
from walk.im2tswalk import TSWalk
from walk.im2randomwalk import RandomWalk
from walk.im2infowalk import InfoWalk
import pickle 
from typing import Union 
from random import choice, shuffle 
import warnings
from utils.preprocess import cutEdges
from copy import deepcopy
import h5py


def im2events(img: Union[int,np.array], walk = 'random', preprocess = True, nsteps = 40, paused: list = [],
                    pos_thres = 0.4, neg_thres = 0.4, fps = 50, 
                    refrac_pd = 0.0, leak_rate_hz = 0., shot_noise_rate_hz = 0.,
                     # traj_preset for inputting trajectory, start_pos for restarting walk eg calling for more timesteps
                     # vec for im2line ie nsteps in the "N" direction 
                    traj_preset = [], start_pos = [], vec:str = None, 
                    frame_h = CAMERA_RES[0], frame_w = CAMERA_RES[1],
                    eps = 0.03, ts_w = 50, ts_mu = 500, ucb_w = 10,
                    maximize = True, warmup = False, warmup_rounds = 2,
                    save = False, test_data = False, bg_fill = 0):
    
    
    assert not vec or not traj_preset

    if vec:
        traj_preset = [vec] * nsteps

    if len(traj_preset) > 0:
            nsteps = len(traj_preset) 
    events = []
    n_events = []

    if type(img) == np.ndarray:
        # to ensure img does not pass frame boundary 
        # im_shape = max(img.shape[0],img.shape[1])
        im_shape = img.shape
    else:
        im_shape = (IM_SIZE,IM_SIZE)

    
    if walk == 'random':
        traj_path = RAND_TRAJPATH if not test_data else RAND_Test_TRAJPATH
        events_path = RAND_EVENTSDIR if not test_data else RAND_Test_EVENTSDIR
        hits_path = RAND_HITSDIR if not test_data else RAND_Test_HITSDIR
        # re sensor_size, event emulator will greyscale any rgb photo 
        walker = RandomWalk(sensor_size= (frame_h,frame_w), im_size = im_shape, start_pos = start_pos)

    elif walk == 'info':
        traj_path = INFO_TRAJPATH 
        events_path = INFO_EVENTSDIR
        hits_path = INFO_HITSDIR
        walker = InfoWalk(sensor_size=(frame_h,frame_w), im_size = im_shape, start_pos = start_pos)

    elif walk == 'eps':
        # TODO new paths if save
        traj_path = INFO_TRAJPATH
        events_path = INFO_EVENTSDIR
        hits_path = INFO_HITSDIR
        walker = EPSWalk(sensor_size=(frame_h,frame_w), im_size = im_shape, maximize = maximize, start_pos = start_pos, eps = eps )
        # warmup = True
        warmup_set = deepcopy(walker.stepset) * warmup_rounds

    elif walk == 'ucb':
        traj_path = INFO_TRAJPATH
        events_path = INFO_EVENTSDIR
        hits_path = INFO_HITSDIR
        walker = UCBWalk(sensor_size=(frame_h,frame_w), im_size=im_shape, maximize=maximize, start_pos = start_pos, w = ucb_w)
        warmup = True
        warmup_set = deepcopy(walker.stepset) * warmup_rounds

    elif walk == 'ts':
        traj_path = TS_TRAJPATH if not test_data else TS_Test_TRAJPATH
        events_path = TS_EVENTSDIR if not test_data else TS_Test_EVENTSDIR
        hits_path = TS_HITSDIR if not test_data else TS_Test_HITSDIR
        walker = TSWalk(sensor_size=(frame_h,frame_w), im_size=im_shape, maximize=maximize, start_pos = start_pos, cdp = ts_w, mu = ts_mu)
        # warmup = True
        warmup_set = deepcopy(walker.stepset) * warmup_rounds

    # placeholder for multiple stabs at infotaxis 
    else:
        raise NotImplementedError
    

    
    if vec is not None:
        assert vec in walker.stepset
    
    imix = img if type(img) == int else "0000"
    events_dir = events_path if save else None
    events_h5 = f"Im_{imix}.h5" if save else None

    v2ee = StatefulEmulator(output_folder=events_dir, # mod from events_dir  
                            dvs_h5= events_h5,  # dont save in emulator because flattens
                            # dvs_h5 = None,
                            num_frames = nsteps,
                            fps = fps,
                            # up from 0.2 default
                            pos_thres = pos_thres,
                            neg_thres = neg_thres,
                            leak_rate_hz=leak_rate_hz,
                            shot_noise_rate_hz= shot_noise_rate_hz,
                            # each frame is 0.02s
                            refractory_period_s=refrac_pd,
                            im_size = im_shape,
                            frame_h = frame_h,
                            frame_w = frame_w, 
                            test_data= test_data,
                            bg_fill=bg_fill)
    
    assert walker.sensor_size[0] == v2ee.frame_hw[0]
    assert walker.sensor_size[1] == v2ee.frame_hw[1]
    assert walker.im_size == v2ee.im_size

    # initialize first frame of the emulator
    # so as not to override EventEmulator.reset()
    v2ee.reset_(img = imix) 

    # includes random steps start in nsteps 
    start = len(walker.walk) - 1 # mod to -1 for full 40 steps ValueError("Illegal value in chunk tuple")
    prev_coords = walker.walk[-1]
    for step in range(start,nsteps):
        if warmup and len(warmup_set) > 0:
            shuffle(warmup_set)
            vec = warmup_set.pop()

        elif traj_preset:
            vec = traj_preset[step - start]

        else:
            vec = None

        next_coords = walker.coord_move(vec=vec)
         # in the event of traj_preset and info walk, will overwrite first moves with random steps
        # next_coords = walker.coord_move(vec=traj_preset[step - start]) if traj_preset else walker.coord_move()

        # in the event of traj_preset and info walk, will overwrite first moves with random steps

        if paused and step in paused:
            raw_events = v2ee.step(coords = next_coords)
        else:
            raw_events = v2ee.step(coords = next_coords)

        # need to do it step by step for 
        if preprocess:
            step_events, n_step_events = cutEdges(x = [raw_events], imtraj=[prev_coords,next_coords])
        else:
            step_events, n_step_events = raw_events, len(raw_events)        # step_events returns [] => len == 0 if step_events is None 
        
        walker.update(next_coords, len(step_events))
        events.append(step_events)
        n_events.append(n_step_events)

        prev_coords = next_coords 

    # closes dvs file if saving etc
    v2ee.cleanup()

    # walk = walker.walk if not traj_preset else traj_preset
    # walk = walker.walk

    

    if save:

        if os.path.isfile(hits_path+f"/Im_{imix}.pkl"):
            # raise Exception(f"{hits_path}/Im_{imix}.pkl file exists")
            os.remove(hits_path+f"/Im_{imix}.pkl")
            print(f"Im {walk} {imix} hits pkl file exists. Replacing file")
        hits_dir = hits_path if save else None
        hits_pkl = f"Im_{imix}.pkl" if save else None
        with open(os.path.join(hits_dir,hits_pkl), 'wb') as fp:
            pickle.dump(n_events, fp)

        if os.path.isfile(traj_path+f"/Im_{imix}.pkl"):
            # raise Exception(f"{traj_path}/Im_{imix}.pkl file exists")
            os.remove(traj_path+f"/Im_{imix}.pkl")
            print(f"Im {walk} {imix} walk traj pkl file exists. Replacing file")
        traj_dir = traj_path if save else None
        traj_pkl = f"Im_{imix}.pkl" if save else None
        with open(os.path.join(traj_dir,traj_pkl), 'wb') as fp:
            pickle.dump(walker.walk, fp)


    # return events, n_events, walker.walk, walker.mu_std # MOD FOR TESTING 
    return events, n_events, walker.walk



from tqdm import tqdm
if __name__ == "__main__":
    # imix = 42
    
    bg = 0 # black
    leak = 0.0 # default is .1 
    save = True
    walk = 'ts'
    testdata = True
    dataset = CIFAR if not save else CIFAR_test

    refrac_pd = 0.0
    threshold = 0.4
    n_steps = 60
    fps = 50
    start_pos = []
    frame_h, frame_w = 96,96

    for imix in tqdm(range(len(dataset))):
        events, nevents, imtraj = im2events(img=imix, walk = walk, nsteps=n_steps, 
                                pos_thres=threshold,neg_thres=threshold, 
                                refrac_pd=refrac_pd,fps=fps, # default noise 
                                test_data= testdata, leak_rate_hz= leak, # defaul leak rate
                                # to restart walk 
                                start_pos = start_pos, 
                                frame_h = frame_h, frame_w = frame_w, preprocess= True, save=save,
                                bg_fill= bg)
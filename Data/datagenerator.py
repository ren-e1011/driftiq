from configs.envar import CAMERA_RES, IM_SIZE, RAND_TRAJPATH, RAND_EVENTSDIR, INFO_TRAJPATH, INFO_EVENTSDIR
import os
import numpy as np
from Data.frames2events_emulator import StatefulEmulator
from walk.im2infowalk import InfoWalk
from walk.im2randomwalk import RandomWalk
import pickle 
from typing import Union 
from random import choice
import warnings
from utils.preprocess import cutEdges


def im2events(img: Union[int,np.array], walk = 'random', nsteps = 40, paused: list = [],
                    pos_thres = 0.4, neg_thres = 0.4, refrac_pd = 0.0, fps = 50, 
                    # traj_preset for inputting trajectory, start_pos for restarting walk eg calling for more timesteps
                    # vec for im2line ie nsteps in the "N" direction 
                     traj_preset = [], start_pos = [], vec:str = None, 
                     frame_h = CAMERA_RES[0], frame_w = CAMERA_RES[1],
                     save = False):
    
    
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
        traj_path = RAND_TRAJPATH
        events_path = RAND_EVENTSDIR
        # re sensor_size, event emulator will greyscale any rgb photo 
        walker = RandomWalk(sensor_size= (frame_h,frame_w), im_size = im_shape, start_pos = start_pos)

    elif walk == 'info':
        traj_path = INFO_TRAJPATH
        events_path = INFO_EVENTSDIR
        walker = InfoWalk(sensor_size=(frame_h,frame_w), im_size = im_shape, start_pos = start_pos)

    # placeholder for multiple stabs at infotaxis 
    else:
        raise NotImplementedError
    
    if vec is not None:
        assert vec in walker.stepset
    
    imix = img if type(img) == int else "0000"
    events_dir = events_path if save else None
    events_h5 = f"Im_{imix}.h5" if save else None

    v2ee = StatefulEmulator(output_folder=events_dir, 
                            dvs_h5= events_h5, 
                            num_frames = nsteps,
                            fps = fps,
                            # up from 0.2 default
                            pos_thres = pos_thres,
                            neg_thres = neg_thres,
                            # each frame is 0.02s
                            refractory_period_s=refrac_pd,
                            im_size = im_shape,
                            frame_h = frame_h,
                            frame_w = frame_w)
    
    assert walker.sensor_size[0] == v2ee.frame_hw[0]
    assert walker.sensor_size[1] == v2ee.frame_hw[1]
    assert walker.im_size == v2ee.im_size

    # initialize first frame of the emulator
    # so as not to override EventEmulator.reset()
    v2ee.reset_(img = img) 

    
    # initialize mean spikes 
    if walk == 'info':

        raw_spikes = []

        round = 1 
        # warmup 
        for vec in walker.stepset * round:
            coords, _ = walker.coord_move(vec=vec)
            raw_spikes.append(v2ee.step(coords=coords))

            # mean_spikes.append(len(v2ee.step(coords=coords)))
            # To save random start in walk
            # If want prior without any record, delete walker.update()
            # if do not want to save random walk start, need to save step coords to cut
            walker.update(x_a_next=coords)

        # take spikes from the second set only after some warmup
        raw_spikes = raw_spikes[-len(walker.stepset):]
        imtraj = walker.walk[-len(walker.stepset) - 1:]
        # list of spike counts 
        # imtraj = imtraj instead of walker.walk
        _, spikes = cutEdges(x=raw_spikes,imtraj=imtraj)
        # initial hit. nhits last step morally equivalent to taking another step 
        # vec = choice(walker.stepset)

        max_spikes = max(max(spikes),1)
        std_spikes = np.std(spikes)
        mean_spikes = int(np.mean(spikes))


        # mean_spikes = sum(mean_spikes)
        # mean_spikes /= len(walker.stepset)
        # mean_spikes = max(int(mean_spikes),1) 

        walker._init_params(mean_spikes, max_spikes, std_spikes)


    # includes random steps start in nsteps 
    start = len(walker.walk)
    prev_coords = walker.walk[-1]
    for step in range(start,nsteps):
        # in the event of traj_preset and info walk, will overwrite first moves with random steps
        next_coords = walker.coord_move(vec=traj_preset[step - start]) if traj_preset else walker.coord_move()

        if paused and step in paused:
            raw_events = v2ee.step(coords = next_coords)
        else:
            raw_events = v2ee.step(coords = next_coords)

        # need to do it step by step for 
        step_events, n_step_events = cutEdges(x = [raw_events], imtraj=[prev_coords,next_coords])
        # step_events returns [] => len == 0 if step_events is None 
        walker.update(next_coords, len(step_events))
        events.append(step_events)
        n_events.append(n_step_events)

        prev_coords = next_coords 

    # closes dvs file if saving etc
    v2ee.cleanup()

    # walk = walker.walk if not traj_preset else traj_preset
    walk = walker.walk

    if save:
        traj_dir = traj_path if save else None
        traj_pkl = f"Im_{imix}.pkl" if save else None
        with open(os.path.join(traj_dir,traj_pkl), 'wb') as fp:
            pickle.dump(walk, fp)

    return events, n_events, walk 
 

if __name__ == "__main__":
    imix = 42
    refrac_pd = 0.0
    threshold = 0.4

    nsteps = 40
    events, nevents, traj = im2events(img=imix,refrac_pd=refrac_pd, pos_thres=threshold, neg_thres=threshold, 
                                 nsteps=40,
                                 frame_h = CAMERA_RES[0],frame_w = CAMERA_RES[1],
                                 walk='info')
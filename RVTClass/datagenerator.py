from envar import * 
from frames2events_emulator import StatefulEmulator
from im2infowalk import InfoWalk
from im2randomwalk import RandomWalk
import pickle 
from typing import Union 


def im2events(img: Union[int,np.array], walk = 'random', nsteps = 40, paused: list = [],
                    pos_thres = 0.4, neg_thres = 0.4, refrac_pd = 0.0, fps = 50, 
                    # traj_preset for inputting trajectory, start_pos for restarting walk eg calling for more timesteps
                    # vec for im2line ie nsteps in the "N" direction 
                     traj_preset = [], start_pos = [], vec:str = None, 
                     frame_h = CAMERA_RES[0], frame_w = CAMERA_RES[1],
                     save = False):
    
    assert vec in {'N','E','S','W', 'NE', 'NW', 'SE', 'SW', None}
    events = []
    # imix may be unnecessary if pass in img

    if type(img) == np.ndarray:
        # assert im_shape[0] == im_shape[1]
        # to ensure img does not pass boundary 
        im_shape = max(img.shape[0],img.shape[1])
    else:
        im_shape = IM_SIZE 
    
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

    if len(traj_preset) > 0:
        nsteps = len(traj_preset) 


    for step in range(nsteps):
        coords = traj_preset[step] if traj_preset else []
        if paused and step in paused:
            step_events = v2ee.step(img=img,coords=traj_preset,walker=walker, vec='X')
        else:
            step_events = v2ee.step(img=img,coords=traj_preset,walker=walker, vec=vec)
        # returns [] if step_events is None 
        events.append(step_events)

    # closes dvs file if saving etc
    v2ee.cleanup()

    walk = walker.walk if not traj_preset else traj_preset

    if save:
        traj_dir = traj_path if save else None
        traj_pkl = f"Im_{imix}.pkl" if save else None
        with open(os.path.join(traj_dir,traj_pkl), 'wb') as fp:
            pickle.dump(walk, fp)

    return events, walk 
 


if __name__ == "__main__":
    pass 
# TODO rename file as datagenerator

#import json
#import os
# implied - proc is returned
# import v2e


from envar import * 
from frames2events_emulator import StatefulEmulator
from im2infowalk import InfoWalk
from im2randomwalk import RandomWalk
import pickle 
from copy import deepcopy

from subprocess import Popen

def im2frames2events(imix, walk = 'random', nsteps = 300, fps = 50):
    
    # imix may be unnecessary if pass in img

    if walk == 'random':
        traj_path = RAND_TRAJPATH
        events_path = RAND_EVENTSDIR
        hits_path = RAND_HITSDIR
        walker = RandomWalk()
        walk2events = im2randframes2events

    elif walk == 'info':
        traj_path = INFO_TRAJPATH
        events_path = INFO_EVENTSDIR
        hits_path = INFO_HITSDIR
        walk2events = im2infoframes2events

    elif walk == 'line':
        walk2events = im2linewalk2events
        
    else:
        raise NotImplementedError

    if isinstance(imix,list):
        # TODO mod to h5 
        proc_ixs = [i for i in imix if not os.path.isfile(events_path+f"/Im_{i}.h5")]
            # test 
        
        # mod to change 
        # im2randframes2events returns the command
        # walk2events no longer returns a cmd with line args 
        # procs = [Popen(walk2events(imix,nsteps,fps,traj_dir=traj_path,traj_pkl=f"Im_{imix}.pkl",events_dir =events_path,events_h5=f"Im_{imix}.h5")) for imix in proc_ixs]
        for imix in proc_ixs:
            walk2events(imix,nsteps,fps,traj_dir=traj_path,traj_pkl=f"Im_{imix}.pkl",events_dir =events_path,events_h5=f"Im_{imix}.h5")
        # for p in procs:
        #     p.wait()
    # generate random walk of specific image - dataloader
    elif isinstance(imix, int):
        eventsout_h5 = f"Im_{imix}.h5"
        trajout_pkl = f"Im_{imix}.pkl"
        # proc = Popen(walk2events(imix,nsteps,fps,traj_dir=traj_path,traj_pkl=trajout_pkl,events_dir =events_path,events_h5=eventsout_h5))
        # proc.wait()
        walk2events(imix,nsteps,fps,traj_dir=traj_path,traj_pkl=trajout_pkl,events_dir =events_path,events_h5=eventsout_h5)

    else:
        raise TypeError
    
    
# pass around randEvents_dict for all img walks else just for one img
def im2randframes2events(imix, nsteps,fps,traj_dir,traj_pkl,events_dir,events_h5, hits_dir = RAND_HITSDIR, record_h = False):
    # t_genevents_s = time.time()
    randEvents_dict = {}

    img = np.array(CIFAR[imix][0])

    # testing higher thresholds, longer refractory period (0.005 in 0.02 )
    # resets
    v2ee = StatefulEmulator(output_folder=events_dir, 
                            dvs_h5= events_h5, num_frames = nsteps,
                            fps = fps,
                            # up from 0.2 default
                            pos_thres = 0.4,
                            neg_thres = 0.4,
                            # for a two second durationlen(cut_events[0])
                            # each frame is 0.02s for a total walk-duration of 6s
                            refractory_period_s=0.05)
    # should reset 
    walker = RandomWalk()

    new_events = None

    for step in range(nsteps):
        
        last_step = walker.walk[-1].copy() if len(walker.walk) > 0 else walker.x_a.copy()
        
        coords = walker.coord_move(x_a=last_step)
        walker.walk.append(coords)

        # CAMERA RES (192,320,3) to fit 3d img
        frame = np.zeros(walker.sensor_size,dtype=int)
        # flipped img because inputting [y,x]
        frame[coords[0]:coords[0]+ IM_SIZE,coords[1]:coords[1]+IM_SIZE] = img

        # returns xa_next, entropy_a to monitor entropy 
       
        # save events 
        new_events = v2ee.em_frame(frame)
        # only save new_events if they are within scope
        num_events = len(new_events) if new_events is not None else 0 


        if record_h:
            y,x = coords
            # if x,y is not a key, create it 
            randEvents_dict.setdefault(x,{}).setdefault(y,{'nevents':0,'nsteps':0})
            randEvents_dict[x][y]['nevents'] += num_events
            # number of steps which reached this coordinate step
            randEvents_dict[x][y]['nsteps'] += 1
        # save walker.walk - internal NW coordinates for walk reconstruction

    v2ee.cleanup()

        # save events h5   
            # implicit in 
            # v2e.v2ecore.emulator __init__ 313-325, 960 - 965
            
    return events, walker.walk 
    


    with open(os.path.join(traj_dir,traj_pkl), 'wb') as fp:
        pickle.dump(walker.walk, fp)

    if record_h:
        hits_pkl = f"Im_{imix}.pkl"
        with open(os.path.join(hits_dir,hits_pkl),'wb') as fh:
            pickle.dump(randEvents_dict, fh)



    
# called for an info walk
# imix is no longer a list 
# MOD use_prior False
def im2infoframes2events(imix,nsteps,fps,traj_dir,traj_pkl,events_dir,events_h5, maxmean = 0,use_prior = True):

    img = np.array(CIFAR[imix][0])

    # testing higher thresholds, longer refractory period (0.005 in 0.02 )
    # resets
    v2ee = StatefulEmulator(output_folder=events_dir, 
                            dvs_h5= events_h5, num_frames = nsteps,
                            fps = fps,
                            # up from 0.2 default
                            pos_thres = 0.4,
                            neg_thres = 0.4,
                            # for a two second duration
                            # each frame is 0.02s for a total walk-duration of 6s
                            refractory_period_s=0.05)
    # should reset 
    # add in p_prior 
    if use_prior:
        with open(os.path.join(RAND_HITSDIR,'priorhits.pkl'),'rb') as f:
            prior = pickle.load(f)
            # prior saved as (320,192) but walk is on (192,320)
            prior = prior.T
    else:
        prior = None

    if maxmean == 0:
        with open(os.path.join(RAND_HITSDIR,'meanhits.pkl'), 'rb') as f:
            meanhits = pickle.load(f)
            meanhits = meanhits.astype(int)
            meanhits = meanhits.T
            #maxmean = meanhits.max()
    
    walker = InfoWalk(p_prior = prior, mean_spikes = meanhits)

    new_events = None
    # while  i not in self.v2ee[imix]['step'].keys():
    # stepset = ['N','E','S','W','NE', 'NW', 'SE', 'SW']

    while new_events is None:
        # i += 1
        
        # for first step 
        last_step = walker.walk[-1].copy() if len(walker.walk) > 0 else walker.x_a.copy()
        # in-place
        
        coords = walker.coord_move(x_a=last_step)
        walker.walk.append(coords)

        frame = np.zeros(walker.sensor_size,dtype=int)
        # flipped img because inputting [y,x]
        frame[coords[0]:coords[0]+ IM_SIZE,coords[1]:coords[1]+IM_SIZE] = img

        # returns xa_next, entropy_a to monitor entropy 
       
        # save events 
        new_events = v2ee.em_frame(frame)
        
    start = len(walker.walk)
    # walker.x_a = walker.walk[-1].copy()
    for step in (start, nsteps):
        #num_events = len(new_events) if new_events is not None else 0 
        # to prevent divide by zero in bayes update 
        num_events = len(new_events) if new_events is not None else 1
        last_step = walker.walk[-1].copy()

        # returns xa_next, entropy_a to monitor entropy 
        # should mod self.x_a
        walker.next_step(h = num_events, x_a = last_step)
        # save events 
        frame = np.zeros(walker.sensor_size,dtype=int)
        frame[coords[0]:coords[0]+ IM_SIZE,coords[1]:coords[1]+IM_SIZE] = img
        new_events = v2ee.em_frame(frame)

        # save walker.walk - internal NW coordinates for walk reconstruction

    v2ee.cleanup()




        # save events h5   
            # implicit in 
            # v2e.v2ecore.emulator __init__ 313-325, 960 - 965
            
            # 
    with open(os.path.join(traj_dir,traj_pkl), 'wb') as fp:
        pickle.dump(walker.walk, fp)



##########################################
     
# for the specific purpose of walking in a constant line followed by any amount of pause - for data observation 
def im2linewalk2events(i = 0,img = None,nsteps = 60, npause = 40, vec="N", refrac_pd = 0.0, thres = 0.4, fps = FPS, start_pos: list = [], frame_h: int = None, frame_w: int = None):
    
    frame_h = DVS_RES[0] if frame_h is None else frame_h
    frame_w = DVS_RES[1] if frame_w is None else frame_w

    events = []
    spikes_count = []
    img = np.array(CIFAR[i][0]) if img is None else img
    walker = RandomWalk(sensor_size = (frame_h,frame_w)) if len(img.shape) == 2 else RandomWalk(sensor_size = (frame_h,frame_w,3))


    ## equivalent to np.transpose(img,axes=(1,0,2))
    imgT = np.swapaxes(img,0,1)

    # no output folder, no dvs_h5
    v2ee = StatefulEmulator(output_folder = None ,dvs_h5 = None,
                            # where 1 frame: 1 timestep
                            num_frames = nsteps + npause,
                            fps = fps,
                            pos_thres = thres,
                            neg_thres = thres,
                            # for a two second duration
                            # each frame is 0.02s 
                            refractory_period_s=refrac_pd
    )
    
    # mod rm dtype = int
    frame = np.zeros([walker.sensor_size[0],walker.sensor_size[1]]) if len(img.shape) == 2 else np.zeros(walker.sensor_size,dtype=int)
    
    # if no start position, walker.walk is initialized to [[sensor_center]]
    coords = walker.walk[-1].copy() if start_pos == [] else start_pos
    frame[coords[0]:coords[0]+ IM_SIZE,coords[1]:coords[1]+IM_SIZE] = imgT
    
    # to initialize the v2ee. should be none - and not appended to events
    new_events = v2ee.em_frame(frame)

    for step in range(nsteps + npause): 
        if step < nsteps: 
            last_step = walker.walk[-1].copy() 
        
            coords = walker.coord_move(vec=vec)
            # CAMERA RES (X,Y,3) to fit 3d img

            # mod dtype = int
            frame = np.zeros(walker.sensor_size)
            frame[coords[0]:coords[0]+ IM_SIZE,coords[1]:coords[1]+IM_SIZE] = imgT

        else:
            # to append to walker
            coords = walker.coord_move(vec='X')
        # returns xa_next, entropy_a to monitor entropy 
    
        # save events 
        new_events = v2ee.em_frame(frame)
        # only save new_events if they are within scope
        # num_events = len(new_events) if new_events is not None else 0 


        if new_events is not None:
            events.append(new_events)
            spikes_count.append(len(new_events))

        # unnecessary to save empty events because time variable is continuous 
            # save for cutEdges which aligns with each step - will discard during preprocess
        else:
            events.append([])
            spikes_count.append(0)


        # even if zero
        # spikes_count.append(num_events)

    v2ee.cleanup()

    return events, spikes_count, walker.walk

    
## Not to save dataset wholesale but to access data chunk on the fly or continuation of a walk
def im2krandsteps2events(i, timesteps=40,refrac_pd = 0.0, thres=0.4, fps = 50, start_pos: list = []):
    events = []
    # spikes_count = []
    img = np.array(CIFAR[i][0])

    # no output folder, no dvs_h5
    v2ee = StatefulEmulator(output_folder = None ,dvs_h5 = None,
                            # where 1 frame: 1 timestep
                            num_frames = timesteps,
                            fps = fps,
                            pos_thres = thres,
                            neg_thres = thres,
                            # for a two second duration
                            # each frame is 0.02s 
                            refractory_period_s=refrac_pd)
    
    
    walker = RandomWalk() 

    # if no start position, walker.walk is initialized to [[sensor_center]]
    coords = walker.walk[-1].copy() if start_pos == [] else start_pos
    frame = np.zeros(walker.sensor_size,dtype=int)
    frame[coords[0]:coords[0]+ IM_SIZE,coords[1]:coords[1]+IM_SIZE] = img

    # should be none - and not appended to events
    new_events = v2ee.em_frame(frame)

    for step in range(timesteps): 
        last_step = walker.walk[-1].copy() 
    
        coords = walker.coord_move()
        # CAMERA RES (X,Y,3) to fit 3d img
        frame = np.zeros(walker.sensor_size,dtype=int)
        frame[coords[0]:coords[0]+ IM_SIZE,coords[1]:coords[1]+IM_SIZE] = img

        # returns xa_next, entropy_a to monitor entropy 
    
        # save events 
        new_events = v2ee.em_frame(frame)
        # only save new_events if they are within scope
        # num_events = len(new_events) if new_events is not None else 0 


        if new_events is not None:
            events.append(new_events)

        # unnecessary to save empty events because time variable is continuous 
            # save for cutEdges which aligns with each step - will discard during preprocess
        else:
            events.append([])


        # even if zero
        # spikes_count.append(num_events)

    v2ee.cleanup()

    return events, walker.walk

from envar import *
from random import sample
from tqdm import tqdm
from multiprocessing import Pool
import h5py
if __name__ == "__main__":
    # im_ixs = [i for i in range(len(CIFAR)) if f"Im_{i}.h5" not in os.listdir(RAND_EVENTSDIR) ]

   


    # #random.sample - without replacement (random.choices with replacement)
    # # ixs = sorted(sample(im_ixs,10000))
    
    # ixs = sample(im_ixs,2)

    # randeventfiles = os.listdir(RAND_EVENTSDIR)
    # ixs = []
    # for file in randeventfiles:
    #     # "Im_43.h5"
    #     ext = file.split('.')
    #     if ext[1] == 'h5':
    #         ixs.append(int(ext[0].split('_')[1]))

    # fails because one of the files is an ipynb notebook
    # ixs = [int(t.split('_')[1].split('.')[0]) for t in ixs ]
    # re.search("[0-9]+",ixs[0]).group()
    
    # im_ixs = [49996,49997]
    
    imixs = [i for i in range(len(CIFAR))]
    events_path = RAND_EVENTSDIR 
    n_frames =300
    frame_rate_hz =50
    walk = "random"

    traj_path = RAND_TRAJPATH if walk == 'random' else INFO_TRAJPATH


    events = None
    for i in tqdm(imixs):
        try: 
            events = np.array(h5py.File(events_path+f"/Im_{i}.h5")['events'], dtype='int32')

            imtraj_pkl = os.path.join(traj_path,f"Im_{i}.pkl")
            with open(imtraj_pkl, 'rb') as fp:
            # dict of event counts accessed by [x][y]
                imtraj = pickle.load(fp)
            # whether file does not exist or is corrupted
            # OSError or FileNotFoundError 
        except OSError:
            if os.path.isfile(events_path+f"/Im_{i}.h5"):
                os.remove(events_path+f"/Im_{i}.h5")
            if os.path.isfile(traj_path+f"/Im_{i}.pkl"):
                os.remove(traj_path+f"/Im_{i}.pkl")
            im2frames2events(i, walk = walk, nsteps = n_frames, fps = frame_rate_hz)
            # events = np.array(h5py.File(events_path+f"/Im_{i}.h5")['events'], dtype='int32')
            print('Generated', f"Im_{i}.h5")
    # for i in tqdm(imixs):
    # # for i in tqdm(im_ixs):
    #     # print(i)
    #     im2frames2events(imix=i, walk='random')

    # pool = Pool(im2frames2events,imixs)
    
    # works
    # with Pool() as p:
    #     p.map(im2frames2events,imixs)
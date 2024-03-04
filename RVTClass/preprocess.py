import os
from envar import *
import pickle

os.chdir(FILEPATH)
import torch
# default base e
from math import log

from omegaconf import DictConfig, OmegaConf
# i is only for debugging 
def cutEdges(x,imtraj):
    # out_nw = [CAMERA_RES[0]//2 - IM_SIZE//2, CAMERA_RES[1]//2 - IM_SIZE//2]
    out_nw = imtraj[0]
    imtraj = imtraj[1:]

    cut_x = []
    cut_counts = []
    assert(len(imtraj) == len(x))
    for i in range(len(imtraj)):

        x_slice = x[i]
        in_nw = imtraj[i]

        # append an empty list with 0 events
        if len(x_slice) > 0:
            # trajectory, is input as [x,y]
            # events are input as [t,x,y]
            x_slice = x_slice[(x_slice[:,2] != out_nw[0]) & (x_slice[:,2] != in_nw[0])
                            & (x_slice[:,2] != out_nw[0]+IM_SIZE-1) & (x_slice[:,2] != in_nw[0]+IM_SIZE-1)
                            & (x_slice[:,1] != out_nw[1]) & (x_slice[:,1] !=in_nw[1])
                            & (x_slice[:,1] != out_nw[1]+IM_SIZE-1) & (x_slice[:,1] != in_nw[1]+IM_SIZE-1)]
        # if memory intensive
        # _x.extend(x_slice)
        if len(x_slice) > 0:
            cut_x.append(x_slice)
        cut_counts.append(len(x_slice))

        out_nw = in_nw
    # _x = np.array(_x)
        # cut counts for testing 
    return cut_x, cut_counts

# debut
# need to handle if traj is a list sequestered by timesteps or nah
def cutEdges_savedData(i,x):
    import pandas as pd
    # if it is a list, will append timestep by timestep
    _x = []
    ts = 1/FPS
    # by inspection
    timestamps = x[:,0] * 10e-7
    timesteps = np.arange(0.0,6.0 + ts,ts)

    timebins = pd.cut(timestamps,bins=timesteps)
    imtraj_pkl = os.path.join(self.traj_path,f"Im_{i}.pkl")
    with open(imtraj_pkl, 'rb') as fp:
    # dict of event counts accessed by [x][y]
        imtraj = pickle.load(fp)

    # 80 x 144 - starting point of all
    out_nw = [CAMERA_RES[0]//2 - IM_SIZE//2, CAMERA_RES[1]//2 - IM_SIZE//2]
    for i,bracket in enumerate(pd.Series(timestamps).groupby(timebins)):
        ixs = bracket[1].index
        x_slice = x[ixs]

        in_nw = imtraj[i]
        # by inspection, is dim 1 corresponds to nw[1] and dim 2 corresponds to nw[0]
        x_slice = x_slice[(x_slice[:,2] != out_nw[0]) & (x_slice[:,2] != in_nw[0])
                            & (x_slice[:,2] != out_nw[0]+IM_SIZE-1) & (x_slice[:,2] != in_nw[0]+IM_SIZE-1)
                            & (x_slice[:,1] != out_nw[1]) & (x_slice[:,1] !=in_nw[1])
                            & (x_slice[:,1] != out_nw[1]+IM_SIZE-1) & (x_slice[:,1] != in_nw[1]+IM_SIZE-1)]

        # if memory intensive
        _x.extend(x_slice)

        out_nw = in_nw
    _x = np.array(_x)
    return _x

# bins = nbins # T


# not relevant for our data setup which by default mixes the polarity
def merge_channel_and_bins(representation: torch.Tensor):
    assert representation.dim() == 4
    # reliable representation?
    return torch.reshape(representation, (-1, height, width))

# def get_shape():
#     return 2 * bins, height, width

# from RVT/data/utils/representations.py > StackedHistogram
    # called by RVT/scripts/genx/preprocess_dataset.py > StackedHistogramFactory

def is_int_tensor(tensor: torch.Tensor) -> bool:
    return not torch.is_floating_point(tensor) and not torch.is_complex(tensor)

def construct_x(x: torch.Tensor, step_t: float, bins: int, fastmode: bool = True) -> torch.Tensor:
    # assert is_int_tensor(x)


    # by inspection
    # passes in in s
    time = x[:,0]
    pol = x[:,3]
    w = x[:,1]
    h = x[:,2]

    dtype = torch.float
    representation = torch.zeros((bins, height, width),
        dtype=dtype, requires_grad=False)

    if x.numel() == 0:
        return representation

    t0_int = time.flatten()[0].item()
    # img walk end
    t1_int = time.flatten()[-1].item()
    assert t1_int >= t0_int
    t_norm = time - t0_int
    # normalize time delta as percentage of range eg in time vec tensor([0.0000, 0.0667, 0.1333, 0.2000, 0.2667, 0.3333, 0.4000, 0.4667, 0.5333,0.6000, 0.6667, 0.7333, 0.8000, 0.8667, 0.9333, 1.0000], device='cuda:0')
    # t_norm = t_norm / max((t1_int - t0_int), 1)
    t_norm = t_norm / (t1_int - t0_int)
    # multiply by bins to define bin separation
    #tensor([ 0.0000,  0.6667,  1.3333,  2.0000,  2.6667,  3.3333,  4.0000,  4.6667,
    #     5.3333,  6.0000,  6.6667,  7.3333,  8.0000,  8.6667,  9.3333, 10.0000],
    #   device='cuda:0')
    t_norm = t_norm * bins
    # ^ convert to bin number [0,9]
    t_idx = t_norm.floor()
    # which bin/timestep to accumulate
    t_idx = torch.clamp(t_idx, max=bins - 1)


    assert torch.max(h).item() < height
    assert torch.max(w).item() < width

    w = w.int()
    h = h.int()
    t_idx = t_idx.int()

    for i in range(bins):
        if torch.any(t_idx == i):
            # that it spiked...
            # representation[i][w[t_idx==i],h[t_idx==i]] += 1
            # vs number of spikes in the time bin 
            for _w, _h in zip(w[t_idx==i], h[t_idx==i]):
                representation[i][_w,_h] += 1
            


    if not fastmode:
        representation = representation.to(torch.uint8)
    return representation



# x is in [batch_size, timesteps,4]
# t_res - number of time bins to chunk
# t_steps - number of time steps to use
# step_t - time difference (s) between each time step
def construct_x_savedData(x: torch.Tensor, step_t: float, bins: int, fastmode: bool = True) -> torch.Tensor:
    """
        In case of fastmode == True: use uint8 to construct the representation, but could lead to overflow.
        In case of fastmode == False: use int16 to construct the representation, and convert to uint8 after clipping.

        Note: Overflow should not be a big problem because it happens only for hot pixels. In case of overflow,
        the value will just start accumulating from 0 again.
    """


     # MOD:: here, channels is the polarity channel
    # channels = 1
    # x [batch_size,timesteps,4] where last dimension is [timestamp,height,width,polarity]
    # dtype = torch.uint8 if fastmode else torch.int16
    # RuntimeError: Input type (torch.cuda.LongTensor) and weight type (torch.cuda.FloatTensor) should be the same

    # device = x.device
    assert is_int_tensor(x)


    # 10e-7 by inspection - hardcode for all?
    time = x[:,0] * 10e-7

    indices = time < (t_steps * step_t )

    x = x[indices]
    # x = torch.tensor(x).unsqueeze(dim=0)

    # TODO not to flatten the polarity

    # pol = x[:,:,3]
    pol = x[:,3]

    # [N, 4], each row contains [timestamp, x coordinate, y coordinate, sign of event (+1 ON, -1 OFF)] v2e.v2ecore.emulator
    # emulator line 890 - events_curr_iter is 2d array [N,4] with 2nd dimension [t,x,y,p]. N is the number of events from this frame
    # h = x[:,:,1]
    # w = x[:,:,2]

    # w = x[:,:,1]
    w = x[:,1]
    # h = x[:,:,2]
    h = x[:,2]

    # why am i putting the height first in representation here
    # to match RVTClass.data.utils.representations > construct representation
    # representation = torch.zeros((bins, width, height),
    #                             dtype=dtype, device=device, requires_grad=False)
    # should move to device automatically - test
    # representation = torch.zeros((bins, height, width),
    #     dtype=dtype, device=device, requires_grad=False)
    dtype = torch.float
    representation = torch.zeros((bins, height, width),
        dtype=dtype, requires_grad=False)



    # assert x.numel() == y.numel() == pol.numel() == time.numel()
    # no spikes
    if x.numel() == 0:
        return representation
        # merge_channel_and_bins(representation.to(torch.uint8))

    time = time[indices]
    # time is the vector of timestamps
    # NOTE: assumes sorted time
    # img walk start
    t0_int = time.flatten()[0].item()
    # img walk end
    t1_int = time.flatten()[-1].item()
    assert t1_int >= t0_int

    # time - tk (vec)
    # range
    t_norm = time - t0_int
    # normalize time delta as percentage of range eg in time vec tensor([0.0000, 0.0667, 0.1333, 0.2000, 0.2667, 0.3333, 0.4000, 0.4667, 0.5333,0.6000, 0.6667, 0.7333, 0.8000, 0.8667, 0.9333, 1.0000], device='cuda:0')
    t_norm = t_norm / max((t1_int - t0_int), 1)

    # multiply by bins to define bin separation
    #tensor([ 0.0000,  0.6667,  1.3333,  2.0000,  2.6667,  3.3333,  4.0000,  4.6667,
    #     5.3333,  6.0000,  6.6667,  7.3333,  8.0000,  8.6667,  9.3333, 10.0000],
    #   device='cuda:0')
    t_norm = t_norm * bins
    # ^ convert to bin number [0,9]
    t_idx = t_norm.floor()
    # which bin/timestep to accumulate
    t_idx = torch.clamp(t_idx, max=bins - 1)


    assert torch.max(h).item() < height
    assert torch.max(w).item() < width


    for i in range(bins):
        if torch.any(t_idx == i):
            representation[i][h[t_idx==i],w[t_idx==i]] += 1

    if not fastmode:
        representation = representation.to(torch.uint8)
    return representation



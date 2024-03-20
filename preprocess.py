from envar import CAMERA_RES, FPS, IM_SIZE, height, width 
import numpy as np

# os.chdir(FILEPATH)
import torch
# default base e
import pandas as pd

def time_chunk_eventstream(x, steps = 40):
    _x = []
    ts = 1/FPS
     # by inspection
    
    start_t = 0.0
    end_t = ts * steps 
    timesteps = np.arange(start_t,end_t + ts,ts)

    timestamps = x[:,0] * 10e-7
    timebins = pd.cut(timestamps,bins=timesteps)

    for bracket in pd.Series(timestamps).groupby(timebins):
        ixs = bracket[1].index
        x_slice = x[ixs]
        # append even if [] - cutEdges will rm and important for spike count keeping
        # if len(x_slice) > 0:
        _x.append(x_slice)
    return _x

# from saved - one stream of events
# def cutEdges(x,imtraj, from_saved = False, steps = 40):

#     if from_saved:
#         x = time_chunk_eventstream(x,steps)

#     # out_nw = CENTER
#     out_nw = imtraj[0]
#     imtraj = imtraj[1:]

#     cut_x = []
#     count_x = []

#     assert(len(imtraj) == len(x))
#     for i in range(len(imtraj)):
#         x_slice = x[i]
#         in_nw = imtraj[i]

#         # append an empty list with 0 events
#         if len(x_slice) > 0:
#             # trajectory, is input as [x,y]
#             # (since sensor is h x w but img is transposed to align)
#             # events are input as [t,x,y]
#             x_slice = x_slice[(x_slice[:,2] != out_nw[0]) & (x_slice[:,2] != in_nw[0])
#                             & (x_slice[:,2] != out_nw[0]+IM_SIZE-1) & (x_slice[:,2] != in_nw[0]+IM_SIZE-1)
#                             & (x_slice[:,1] != out_nw[1]) & (x_slice[:,1] !=in_nw[1])
#                             & (x_slice[:,1] != out_nw[1]+IM_SIZE-1) & (x_slice[:,1] != in_nw[1]+IM_SIZE-1)]
#         # after cut edges 
#         if len(x_slice) > 0:
#             cut_x.append(x_slice)
        
#         # even if 0 - for spike count keeping and experimentation 
#         count_x.append(len(x_slice))

#         out_nw = in_nw

#     return cut_x, count_x  

# from saved - one stream of events
def cutEdges(x,imtraj, from_saved = False, steps = 40):

    def cutSlice(x_slice,in_nw,out_nw):
        # append an empty list with 0 events
        if len(x_slice) > 0:
            # trajectory, is input as [x,y]
            # (since sensor is h x w but img is transposed to align)
            # events are input as [t,x,y]
            x_slice = x_slice[(x_slice[:,2] != out_nw[0]) & (x_slice[:,2] != in_nw[0])
                            & (x_slice[:,2] != out_nw[0]+IM_SIZE-1) & (x_slice[:,2] != in_nw[0]+IM_SIZE-1)
                            & (x_slice[:,1] != out_nw[1]) & (x_slice[:,1] !=in_nw[1])
                            & (x_slice[:,1] != out_nw[1]+IM_SIZE-1) & (x_slice[:,1] != in_nw[1]+IM_SIZE-1)]
        return x_slice
    
    if from_saved:
        x = time_chunk_eventstream(x,steps)

    assert len(imtraj) > 1
    # out_nw = CENTER
    out_nw = imtraj[0]
    imtraj = imtraj[1:]
    in_nw = imtraj[0]

    

    assert(len(imtraj) == len(x))

    if len(x) == 1:
        # could be an empty list 
        cut_x = cutSlice(x[0], in_nw, out_nw)
        count_x = len(cut_x)

    else: 
        cut_x = []
        count_x = []
        for i in range(len(imtraj)):
            x_slice = x[i]
            in_nw = imtraj[i]

            x_slice = cutSlice(x_slice, in_nw, out_nw)

            # after cut edges 
            if len(x_slice) > 0:
                cut_x.append(x_slice)
            
            # even if 0 - for spike count keeping and experimentation 
            count_x.append(len(x_slice))

            out_nw = in_nw
    
    return cut_x, count_x


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

# x is in [batch_size, timesteps,4]
def construct_x(x: torch.Tensor, step_t: float, bins: int, fastmode: bool = True, height: int = CAMERA_RES[0], width: int = CAMERA_RES[1]) -> torch.Tensor:
    """
        In case of fastmode == True: use uint8 to construct the representation, but could lead to overflow.
        In case of fastmode == False: use int16 to construct the representation, and convert to uint8 after clipping.

        Note: Overflow should not be a big problem because it happens only for hot pixels. In case of overflow,
        the value will just start accumulating from 0 again.
    """

    # assert is_int_tensor(x)
    # in s
    time = x[:,0]
    pol = x[:,3]
    w = x[:,1]
    h = x[:,2]

    dtype = torch.float
    # envar.height, envar.width
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

    # return number of spikes
    for i in range(bins):
        if torch.any(t_idx == i):
            # that it spiked...
            # representation[i][w[t_idx==i],h[t_idx==i]] += 1
            # vs number of spikes in the time bin 
            for _w, _h in zip(w[t_idx==i], h[t_idx==i]):
                representation[i][_w,_h] += 1
    
    # return 1 if spiked 
    # for i in range(bins):
    #     if torch.any(t_idx == i):
    #         representation[i][h[t_idx==i],w[t_idx==i]] += 1

    if not fastmode:
        representation = representation.to(torch.uint8)
    return representation




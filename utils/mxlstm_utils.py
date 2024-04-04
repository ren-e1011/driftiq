# ChatGPT translation of matrixlstm_helpers_kernel.cu which is not compiling - with some edits by yt 

import torch
import torch.nn as nn
import torch.nn.functional as F
from pandas import Series
from itertools import chain

# cuts the padded events or event coordinates to the counts of events recorded
# presumes lengths of batch_size * timesteps * values
# returns shape of batch_size * timesteps * values
# if want batch_size by values need to concat
def strip_events(events, lengths):

    batch_size = len(lengths)
    assert len(events) == batch_size
    shorn_events = [[] ] * batch_size
    
    for i, lenlist in enumerate(lengths):
# tolist for tensor items as list
        # shorn_events[i] = [events[i][j * nsteps: j * nsteps + l].tolist() for j,l in enumerate(lenlist)] # item not to keep tensor, grad 
        # shorn_events[i] += [events[i][j][:l].tolist() for j,l in enumerate(lenlist)]
        shorn_events[i] = events[i][:sum(lenlist)].tolist()
    return shorn_events

# events is in [batch_size, ]
# lengths are a list of event counts in each timestep across stimulus across all samples in batch
def nevents_x_coord(coords, lengths, out_w, out_h):
    # todo more efficient way to do this 
   
    evxxy= torch.zeros([len(coords), out_w,out_h])
    # for i in lengths:
    #     events_i = [events[i][:l] for l in lengths[i]]
    
    shorn_coords = strip_events(list(coords),lengths)
    # there has got to be a more elegant way to do this 
    for i,batch_i in enumerate(shorn_coords):
        coord_xy_list = shorn_coords[i]
        # for coord_xy_list in batch_i:
        for x,y in coord_xy_list:
            evxxy[i,x,y] += 1
            # evxxy[batch_i][coord_xy_list] += 1
            # coord_xy = Series(coord_xy).value_counts()
            # for coord in coords:
            #     evxxy[coord] = coords[coord]
    
    return evxxy

def group_rf_bounded_overlap_gpu():
    return

def n_interval_events_gpu():
    return

def intervals_to_batch_gpu():
    return 

def group_rf_gpu():
    return


def flat_3to2dlist(mx):
    return [list(chain(*mx[i])) for i in range(len(mx))]

# number of events in each receptive field of every sample 
# also replaces n_rf_events_cuda_kernel
def n_rf_events_cuda(rf_idx, num_rf):
    
    batch_size = len(rf_idx) # rf_idx.size(0)
    # event_size = rf_idx.size(1)

    # shape_y = lengths.shape[1]

    # moved to group_rf_gpu
    # if nevents_x_step: # if not None
    #     rf_idx = strip_events(rf_idx,nevents_x_step) # ** so as not to confuse padded events for events at 0,0
    #     # sum([len(rf) for rf in rf_idx]) == lengths.sum()
    #     rf_idx = flat_3to2dlist(rf_idx)

    # Create a tensor to hold output counts
    rf_events_count = torch.zeros(batch_size, num_rf)

    # step_size = rf_idx // num_rf

    # rf_idx in batch_size * nsteps 
    for batch_i in range(batch_size):
        rf_coordlist = rf_idx[batch_i]
        # len(rf_coordlist) == sum(nevents_x_step[batch_i])
        for rf_i in rf_coordlist:
            rf_ = int(rf_i) # rf_i.item()
            # rf_events_count[batch_i,rf_i] = sum(lengths[batch_i][rf_i:rf_i:step_size])
            # x,y = rf_ % shape_y, rf_ // shape_y
            # rf_events_count[batch_i][rf_] += int(lengths[batch_i,x,y])
            rf_events_count[batch_i][rf_] += 1

    # # ChatGPT translation of matrix_helpers_kernel.cu n_rf_events_cuda_kernel
    # for batch_id in range(batch_size):
    #     for e in range(event_size):
    #         if e >= lengths[batch_id]:
    #             break
    #         read_index = batch_id * event_size + e
    #         rf_id = rf_idx[read_index].item()  # Assuming rf_idx is a PyTorch tensor
    #         if rf_id == -1:
    #             continue
    #         write_index = batch_id * num_rf + rf_id
    #         rf_events_count[write_index] += 1



    return rf_events_count


    
def min2d_scalar_cuda(input, scalar):
    N = input.size(0) # batch_size 
    M = input.size(1)

    # Create the output tensor
    output = torch.zeros_like(input)

    # # Define the number of threads per block
    # threadsPerBlock = (32, 32)
    # # Calculate the number of blocks
    # numBlocks = ((N + threadsPerBlock[0] - 1) // threadsPerBlock[0],
    #              (M + threadsPerBlock[1] - 1) // threadsPerBlock[1])

    
    for i in range(N):
        for j in range(M):
            output[i][j] = min(input[i][j].item(), scalar)

    # kernel(input, scalar.cpu().item().to(input.dtype), output, N, M)
    
    return output

def group_rf_bounded_cuda_kernel(features, rf_ids, nevents_x_step,
                                     rf_offsets, rf_lengths, groups, 
                                     gr_last_id, gr_batch_id, 
                                     gr_h, gr_w, out_w, 
                                     batch_size, event_size, 
                                     feature_size, num_rf,
                                     new_batch_size, max_rf_events, bound_max):
    num_rf_finished = 0
    batch_size = len(nevents_x_step) # MOD

    for batch_id in range(batch_size):
    # if batch_id < batch_size:
        # for e in range(lengths[batch_id] - 1, -1, -1):
        read_offset = 0 
        for e in range(len(rf_ids[batch_id])-1,-1,-1): # len(rf_ids[batch_id]) == sum(nevents_x_step[batch_id])
            # rf_id = rf_ids[batch_id * event_size + e].item() # event-size is sketchy 
            # + prev e 

            # event pixel/rf 
            rf_id = int(rf_ids[batch_id][e]) # mod from .item() because sending in list 

            # so is this their way of cutting them...they originally padded with -1...clever
            # if rf_id == -1:
            #     continue
            
            # rf_len = rf_lengths[batch_id * num_rf + rf_id].item()

            # validated by uniq,counts = unique(rf_ids[batch_id],return_counts=True) and counts[np.where(uniq == rf_id)] == rf_len
            # number of events at loc rf over the sample 
            rf_len = int(rf_lengths[batch_id][rf_id])
            
            # filling in events ev_step: last_write_pos
            ev_step = 1
            last_write_pos = rf_len - 1

            if bound_max and (max_rf_events < rf_len):
                ev_step = rf_len // (max_rf_events - 1)
                last_write_pos = max_rf_events - 1
            
            # rf_pos = rf_offsets[batch_id * num_rf + rf_id].item() - 1

            # position of the receptive field within the output tensor 
            # rf_offsets of size [batch_size, 9216]
            rf_pos = int(rf_offsets[batch_id][rf_id]) - 1
            
            if gr_last_id[rf_pos].item() < rf_len:
                # Retrieve the number of events already placed inside the target rf. -1 for indexing 
                event_pos = rf_len - 1 - gr_last_id[rf_pos].item()
                
                write_event_pos = int(last_write_pos - (gr_last_id[rf_pos].item() // ev_step))
                prev_write_event_pos = last_write_pos - max(0, (gr_last_id[rf_pos].item() - 1) // ev_step)
                
                if gr_last_id[rf_pos].item() + 1 == rf_len:
                    write_event_pos = 0
                
                if write_event_pos >= 0 and (write_event_pos != prev_write_event_pos 
                                             or gr_last_id[rf_pos].item() == 0 
                                             or gr_last_id[rf_pos].item() + 1 == rf_len):
                    # write_offset = write_event_pos * (new_batch_size * feature_size) + rf_pos * feature_size
                    # write_offset = int(write_offset)

                    # read_offset = batch_id * (event_size * feature_size) + e * feature_size
                    # event_size = nevents_x_step[batch_id][e]
                    
                    # # groups[write_offset:write_offset + feature_size] = features[read_offset:read_offset + feature_size]
                    # groups[write_offset:write_offset + feature_size] = features[batch_id][read_offset:read_offset + feature_size]

                    groups[write_event_pos][rf_pos][:] = features[batch_id][e][:]
                
                gr_last_id[rf_pos] += 1
                
                if gr_last_id[rf_pos].item() == rf_len:
                    num_rf_finished += 1
                
                if event_pos == 0:
                    gr_batch_id[rf_pos] = batch_id
                    gr_h[rf_pos] = rf_id // out_w
                    gr_w[rf_pos] = rf_id % out_w

            read_offset += event_size 
        
        # We incremented gr_last_id each time even if a write was not actually
        # performed due to ev_step > 1. Here we fix gr_last_id replacing the
        # actual length for the receptive fields having more events
        for rf_id in range(num_rf):
            rf_pos = rf_offsets[batch_id][rf_id]
            rf_len = rf_lengths[batch_id][rf_id]
            if bound_max and max_rf_events < rf_len:
                gr_last_id[rf_pos] = max_rf_events


# mod from matrixlstm_helpers which calls group_rf_bounded_cuda in matrixlstm_helpers_kernel.cu which does not run
def group_rf_bounded_gpu(input, ids, nevents_x_step,
                                    max_events_per_rf,
                                    output_shape_w,
                                    output_shape_h,
                                    keep_most_recent, bound_max: bool = False):
    
    batch_size = input.shape[0] 
    event_size = input.shape[1]
    feature_size = input.shape[2]
    max_rf_events = int(max_events_per_rf.item()) # ignore ? 

    w,h = output_shape_w, output_shape_h ##lengths.shape[1], lengths.shape[2]
     # MOD
    if nevents_x_step: # if not None
        rf_ids = strip_events(ids,nevents_x_step) # ** so as not to confuse padded events for events at 0,0
        # sum([len(rf) for rf in rf_idx]) == lengths.sum()
        # rf_ids = flat_3to2dlist(rf_ids)
    else:
        rf_ids = ids

    # number of events in each receptive field of every sample 
    # rf_counts = n_rf_events_cuda(ids, lengths, nevents_x_step, h*w)
    rf_counts = n_rf_events_cuda(rf_ids, h*w)

    if bound_max and max_rf_events > 0:
        bounded_rf_counts = min2d_scalar_cuda(rf_counts, max_rf_events)

    else:
        bounded_rf_counts = rf_counts

    # max events in a receptive field - pixel event stream
    new_event_size = int(torch.max(bounded_rf_counts).item())
    # position that each non empty receptive field will have in the output tensor
    # torch .gt Returns A boolean tensor that is True where input is greater than other and False elsewhere
    rf_offsets = torch.cumsum(bounded_rf_counts.gt(0).view(-1), dim=-1).view(bounded_rf_counts.size())
    # new flat batch size (tot num of non empty receptive fields)
    new_batch_size = int(rf_offsets[batch_size - 1, h * w - 1].item())

    # Create a tensor to hold output groups
    groups = torch.zeros(new_event_size, new_batch_size, feature_size)
    gr_last_id = torch.zeros(new_batch_size, dtype=torch.long)
    gr_batch_id = torch.zeros(new_batch_size, dtype=torch.long)
    gr_h = torch.zeros(new_batch_size, dtype=torch.long)
    gr_w = torch.zeros(new_batch_size, dtype=torch.long)

    # Allocate a thread for each sample, each one will process all the events
    # in the sample (just the batch loop is parallelized)
    # threadsPerBlock = 32
    # numBlocks = (batch_size + threadsPerBlock - 1) // threadsPerBlock

    # We have two strategies based on the value of keep_most_recent
    # - keep_most_recent is True: we pass to group_rf_bounded_cuda_kernel the
    #      rf_counts bounded to max_rf_events. This way the grouping algorithm
    #      will only look at the last max_rf_events (or less) events in each
    #      receptive field, copying each of them (step = 1) in the output
    # - keep_most_recent is False: we pass to group_rf_bounded_cuda_kernel the
    #      NOT bounded version of rf_counts so that the algorithm will look at
    #      all the events in the receptive field. However, it will write with
    #      a step > 1, not copying some of the events in the output tensor.
    #      We always compute new_event_size using the bounded version!
    if keep_most_recent:
        rf_counts = bounded_rf_counts

    
   

    group_rf_bounded_cuda_kernel(
            input, rf_ids, nevents_x_step,
            rf_offsets, rf_counts, groups, 
            gr_last_id, gr_batch_id, 
            gr_h, gr_w, w, 
            batch_size, event_size, 
            feature_size, h * w, 
            new_batch_size, max_rf_events, bound_max)

    # Decrement all values by one (we want the id, not the count)
    gr_last_id -= 1

    return gr_batch_id, gr_last_id, gr_h, gr_w, groups



    # input = input[ids]
    # events = [[None] * output_shape[0]] * output_shape[1]

    # for batch_i,sample in enumerate(input):
    #     events_lens = lengths[batch_i] # list of len(timesteps)
    #     sample_events = [ev[:l,:] for ev, l in zip(sample,events_lens)]
    #     if not events

    # return gr_batch_id, gr_last_id, rel_gr_h, rel_gr_w, batch_groups
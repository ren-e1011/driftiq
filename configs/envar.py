import os
 
# which of these are necessary or useful 
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# from RVT > train.py
# see https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES']='2, 3'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# https://freedium.cfd/https://iamholumeedey007.medium.com/memory-management-using-pytorch-cuda-alloc-conf-dabe7adec130
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1"

# height, width 

# SEQ_LEN = 1600 # number of events, not length of video 
# # from RVT/scripts/genx/preprocess_dataset.py > process_sequence 
# ts_step_ev_repr_ms = 50

# height = CAMERA_RES[0]
# width = CAMERA_RES[1]

# EPSILON = 1e-6

# # and config file variables from conf_preprocess/representation/stacked_hist.yaml,conf_preprocess/extraction/const_duration.yaml called by RVT/scripts/genx/README.md > For the Gen1 dataset ''' '''
# NUM_PROCESSES = 20
# ev_repr_num_events = 50 # if config.event_window_extraction.method == AggregationType.COUNT in RVT/scripts/genx/conf_preprocess/extraction/const_duration.yaml (method:DURATION)
# ev_repr_delta_ts_ms = 50 #  if config.event_window_extraction.method == AggregationType.DURATION in RVT/scripts/genx/conf_preprocess/extraction/const_duration.yaml (method:DURATION)

# # RVT/scripts/genx/preprocess_dataset.py > labels_and_ev_repr_timestamps
# ts_step_frame_ms = 100

# # RuntimeError: Given groups=1, weight of size [64, 20, 7, 7], expected input[2, 10, 260, 346] to have 20 channels, but got 10 channels instead:: nbins (T) is set to 10 but because they flatten the polarity which is delivered as 2 channels in theirs and 1 in ours, presume nbins should be 20 
# # ^ 10 bins for each channel? 
# nbins = 20 # T 
# count_cutoff = 10 #...or MISSING?

# #### Unnecessary 
# # preprocess_dataset > process_sequence 
# align_t_ms = 100 
# # preprocess_dataset > process_sequence > labels_and_ev_repr_timestamps
# # 
# # mu-seconds 
# align_t_us = align_t_ms * 1000
# delta_t_us = ts_step_ev_repr_ms * 1000

# COMET_API_KEY = ''

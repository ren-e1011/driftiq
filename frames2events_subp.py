# adapted from https://gist.github.com/duguyue100/f60bdc1eb0e5b51586ca3594d9d72cb7 
# why does ^ random.uniform select from range of parameters
# vs DVS_PARAMS for Clean, Noisy 
"""
Use V2E to simulate the entire dataset.
For Ideal range:
    - threshold: 0.2-0.5
    - sigma: 0.03-0.05
    - cutoff_hz: 0
    - leak_rate_hz: 0
    - shot_noise_rate_hz: 0
For bright light range:
    - threshold: 0.2-0.5
    - sigma: 0.03-0.05
    - cutoff_hz: 80-120
    - leak_rate_hz: 0.1
    - shot_noise_rate_hz: 0.1-5 Hz
For low light range:
    - threshold: 0.2-0.5
    - sigma: 0.03-0.05
    - cutoff_hz: 20-60
    - leak_rate_hz: 0.1
    - shot_noise_rate_hz: 10-30 Hz
Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

import json
import os
# implied - proc is returned
# import v2e

from im2randomwalk import pickleRandomWalk

def im2frames2events(args,imix, overwrite = False):

    out_folder = args.events_path
    in_file = f"{args.video_path}/Im_{imix}.mov"
    out_file_h5 = f"Im_{imix}.h5"
    out_file_avi = f"Im_{imix}.avi"
    output_mode = args.camera_config

    exposure = 1/int(args.frame_rate_hz)

    if not overwrite and os.path.isfile(out_folder+"/"+out_file_avi):
        # test 
        print(out_file_avi,"File exists")
        return
    # TODO should be in dataloader but requires subprocess working conditionally as well as in parallel 
    if args.walk == 'random':

       pickleRandomWalk(args, imix)     
        
    else:
        raise NotImplementedError

    # set configs
    # configs from v2e tutorial
    # Ideal/Clean range
    if args.condition == "Clean":
        # threshold in log_e intensity change to trigger a positive/negative event. (default: 0.2)
        thres = 0.2
        # 1-std deviation threshold variation in log_e intensity change. (default: 0.03)
        sigma = 0.03
        # photoreceptor first-order IIR lowpass filter cutoff-off 3dB frequency in Hz - see https://ieeexplore.ieee.org/document/4444573 (default: 300)
        cutoff_hz = 0
        # leak event rate per pixel in Hz - see https://ieeexplore.ieee.org/abstract/document/7962235 (default: 0.01)
        leak_rate_hz = 0
        # Temporal noise rate of ON+OFF events in darkest parts of scene; reduced in brightest parts.
        shot_noise_rate_hz = 0
    
    # not relevant for this experiment
    elif args.condition == 'Noisy':
        thresh = 0.2
        sigma_thresh = 0.05
        cutoff_hz = 30
        leak_rate_hz = 0.1
        shot_noise_rate = 5


        # get root folder list
 

    v2e_command = [
        "v2e.py",
        # video file or image folder!! 
        #"-i", folder,
        "-i", in_file,
        # output folder
        "-o", out_folder,
        f"--{output_mode}",
        # overwrites files in existing folder (checks existence of non-empty output_folder)
        "--overwrite",
        # seems right for testing. mod to false?
        # if specifying --output_folder, makes unique output folder based on output_folder eg output 1 if non-empty out_put folder already exists
        "--unique_output_folder", "true",
        "--no_preview",
        # "--skip_video_output",
        "--disable_slomo",
        # MOD
        "--dvs_exposure", "duration", "{}".format(exposure),
        # "--dvs_exposure", "0.02",
        "--pos_thres", "{}".format(thres),
        "--neg_thres", "{}".format(thres),
        "--sigma_thres", "{}".format(sigma),
        "--cutoff_hz", "{}".format(cutoff_hz),
        "--leak_rate_hz", "{}".format(leak_rate_hz),
        "--shot_noise_rate_hz", "{}".format(shot_noise_rate_hz),
        "--input_frame_rate", "{}".format(args.frame_rate_hz),
        # "--input_slowmotion_factor", "17.866666708",
        "--input_slowmotion_factor", "1.0",
        "--dvs_h5", out_file_h5,
        "--dvs_vid", out_file_avi,
        # only affects playback rate - for viewing consistency - asthetic only
        "--avi_frame_rate", "{}".format(args.frame_rate_hz),
        "--auto_timestamp_resolution", "false"]

    # subprocess.run(v2e_command)
    return v2e_command
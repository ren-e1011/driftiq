# adapted from https://gist.github.com/duguyue100/f60bdc1eb0e5b51586ca3594d9d72cb7 
# why does ^ random.uniform select from range of parameters
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
import argparse
import os
import glob
import subprocess
import random
import json

parser = argparse.ArgumentParser()

parser.add_argument("--input_path", type=str, default="./Data/RandomImWalk/300Frames50Hz")
parser.add_argument("--output_path", type=str, default="./Data/RandomImWalk/346DVSEvents")
parser.add_argument("--condition", type=str,
                    help="'Clean', 'bright', 'dark'", default="Clean")
# implement select list? 
parser.add_argument("--im_ix", type=list, default=None)
parser.add_argument("--input_frame_rate",type=int,default=50)

args = parser.parse_args()

# set configs
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


    # get root folder list
valid_folders = sorted(
    glob.glob(
        os.path.join(args.data_root, "*", "image_*")))
valid_folders = [x for x in valid_folders if ".npz" not in x]

params_collector = {}

for folder in valid_folders:
    out_filename = os.path.basename(folder)+".h5"
    out_folder = os.path.dirname(folder)
    out_folder = out_folder.replace(args.data_root, args.output_root)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    folder = os.path.join(folder, "images")


    params_collector[os.path.join(out_folder, out_filename)] = {
        "thres": thres,
        "sigma": sigma,
        "cutoff_hz": cutoff_hz,
        "leak_rate_hz": leak_rate_hz,
        "shot_noise_rate_hz": shot_noise_rate_hz}

    # dump bias configs all the time
    with open(os.path.join(args.output_root,
                           "dvs_params_settings.json"), "w") as f:
        json.dump(params_collector, f, indent=4)

    v2e_command = [
        "v2e.py",
        "-i", folder,
        "-o", out_folder,
        "--overwrite",
        "--unique_output_folder", "false",
        "--no_preview",
        "--skip_video_output",
        "--disable_slomo",
        # MOD
        "--dvs_exposure", 1/args.input_frame_rate,
        "--pos_thres", "{}".format(thres),
        "--neg_thres", "{}".format(thres),
        "--sigma_thres", "{}".format(sigma),
        "--cutoff_hz", "{}".format(cutoff_hz),
        "--leak_rate_hz", "{}".format(leak_rate_hz),
        "--shot_noise_rate_hz", "{}".format(shot_noise_rate_hz),
        "--input_frame_rate", args.input_frame_rate,
        # "--input_slowmotion_factor", "17.866666708",
        "--dvs_h5", out_filename,
        "--dvs_aedat2", "None",
        "--dvs_text", "None",
        "--dvs_exposure", "duration", "0.001",
        "--auto_timestamp_resolution", "false"]

    subprocess.run(v2e_command)
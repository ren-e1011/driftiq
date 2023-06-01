import os
# The full documentation is here: https://github.com/SensorsINI/v2e#render-emulated-dvs-events-from-conventional-video
import v2e

input_path = "./Data/RandomImWalk/300Frames50Hz" 
input_frame_rate = 50 

output_path = "./Data/RandomImWalk/346DVSEvents"

import os

class frames2events(object):

    def __init__(self, out_filename,imix = 0, condition="Clean", output_mode = "dvs346"):
        # extra
        super().__init__()

        self.imix = imix
        

        if condition == "Clean":
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
        elif condition == "Noisy":
            thres = 0.2
            sigma_thres = 0.05
            cutoff_hz = 30
            leak_rate_hz = 0.1
            shot_noise_rate_hz = 5

        # 0.025 at 50Hz
        dvs_exposure = f"duration {1/input_frame_rate}"

        # On headless platforms, with no graphics output, use --no_preview option to suppress the OpenCV windows.
        v2e_command = ["v2e"] + ["-i", input_path] + ["-o", output_path] + ["--dvs_h5", out_filename] + ["--dvs_aedat2", "None"] + ["--dvs_text", "None"] + ["--no_preview"] \
        + ["--input_frame_rate", "{}".format(input_frame_rate)] + ["--dvs_exposure", dvs_exposure] + ["--disable_slomo"] + ["--auto_timestamp_resolution", "false"] + [f"--{output_mode}"] \
        + ["--pos_thres", "{}".format(thres)] + ["--neg_thres", "{}".format(thres)] + ["--sigma_thres", "{}".format(sigma)] \
        # DVS non-idealities
        + ["--cutoff_hz", "{}".format(cutoff_hz)] + ["--leak_rate_hz", "{}".format(leak_rate_hz)] + ["--shot_noise_rate_hz", "{}".format(shot_noise_rate_hz)]
        + [f"--{output_mode}"]
        # if the output will rewrite the previous output
        overwrite=True
        if overwrite:
            v2e_command.append("--overwrite")


        self.final_v2e_command = " ".join(v2e_command)


    def vid2events(self):
        if input_path == "" or not os.path.isfile(input_path):
            print("The file path is empty or invalid, choose a file")
        # infile and outfile name - in frames and events folders, respectively
        filename = f"Im{self.imix}"
        h5_outfile = filename + '.h5'
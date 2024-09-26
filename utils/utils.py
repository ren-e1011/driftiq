from pathlib import Path 
import os, sys
sys.path.append(str(Path.cwd().parent)) # for pythonpath 
sys.path.append(str(Path.cwd()))


from random import choice 
# from configs.envar import CAMERA_RES, IM_SIZE
from copy import deepcopy


def _coord_move(vec: str, x_a: list, stepset: list, sensor_size = (96,96), im_size=32):
        # else will modify x_a
        x_a = deepcopy(x_a)
        stepset = deepcopy(stepset)

        if vec == 'N' and (x_a[1] - 1 >= 0):
            x_a[1] -= 1
            # 
            
        elif vec == 'NE' and (x_a[1] - 1 >= 0) and (x_a[0] + im_size < sensor_size[0]):
            x_a[1] -= 1
            x_a[0] += 1

        elif vec == 'E' and (x_a[0] + im_size < sensor_size[0]):
            x_a[0] += 1

        elif vec == 'SE' and (x_a[1] + im_size < sensor_size[1]) and (x_a[0] + im_size < sensor_size[0]):
            x_a[1] += 1
            x_a[0] += 1

        elif vec == 'S' and (x_a[1] + im_size < sensor_size[1]):
            x_a[1] += 1

        elif vec == 'SW' and (x_a[1] + im_size < sensor_size[1]) and (x_a[0] - 1 >= 0):
            x_a[1] += 1
            x_a[0] -= 1

        elif vec == 'W' and (x_a[0] - 1 >= 0):
            x_a[0] -= 1 

        elif vec == 'NW' and (x_a[1] - 1 >= 0) and (x_a[0] - 1 >=0):
            x_a[1] -= 1
            x_a[0] -= 1
            # stand still 
        elif vec == 'X':
            x_a = x_a 
        else: 
            # if edge, pick a random step 
            # would work for removing just one edge and also if the img is eg in a corner there are 3 
            stepset = list(filter(lambda x: x != vec, stepset))
            # vec is not None so will not instantiate new stepset
            
            # x_a, vec = _coord_move(vec=choice(stepset),x_a=x_a,stepset=stepset)
            x_a, vec = _coord_move(vec=choice(stepset),x_a=x_a,stepset=stepset, sensor_size=sensor_size, im_size=im_size)
            # print(f"out of frame, step {step}")import itertools
            
            # coord_move(<- if edge, pick a new )
        # return vec which might be different than 
        return x_a, vec 


def _traj_to_dir(imtraj: list):
        
        dir_list = []

        assert len(imtraj) > 1
        out_nw = imtraj[0]
        imtraj = imtraj[1:]
        in_nw = imtraj[0]

        for i in range(len(imtraj)):
            in_nw = imtraj[i]
            if in_nw == [out_nw[0],out_nw[1]-1]:
                dir_list.append('N')

            elif in_nw == [out_nw[0],out_nw[1]+1]:
                dir_list.append('S')

            elif in_nw == [out_nw[0]-1,out_nw[1]]:
                dir_list.append('W')

            elif in_nw == [out_nw[0]+1,out_nw[1]]:
                dir_list.append('E')
            
            elif in_nw == [out_nw[0]-1,out_nw[1]-1]:
                dir_list.append('NW')

            elif in_nw == [out_nw[0]+1,out_nw[1]-1]:
                dir_list.append('NE')

            elif in_nw == [out_nw[0]-1,out_nw[1]+1]:
                dir_list.append('SW')

            elif in_nw == [out_nw[0]+1,out_nw[1]+1]:
                dir_list.append('SE')

            # in_nw == out_nw
            else:
                dir_list.append('X')
            out_nw = in_nw 
        return dir_list 

if __name__ == "__main__":
    x_list = [[32,32]]
    v_list = []
    for i in range(60):
        x,v = _coord_move(vec="N",x_a=x_list[-1],stepset=['N','S','W','E','NW','NE','SW','SE'])
        x_list.append(x)
        v_list.append(v)
    print(x_list)
    print(v_list)
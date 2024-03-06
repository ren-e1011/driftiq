
from random import choice 
from envar import CAMERA_RES, DVS_RES, IM_SIZE


def _coord_move(vec: str, x_a: list, stepset: list, sensor_size = CAMERA_RES):
        if vec == 'N' and (x_a[1] - 1 >= 0):
            x_a[1] -= 1
            # 
            
        elif vec == 'NE' and (x_a[1] - 1 >= 0) and (x_a[0] + IM_SIZE < sensor_size[0]):
            x_a[1] -= 1
            x_a[0] += 1

        elif vec == 'E' and (x_a[0] + IM_SIZE < sensor_size[0]):
            x_a[0] += 1

        elif vec == 'SE' and (x_a[1] + IM_SIZE < sensor_size[1]) and (x_a[0] + IM_SIZE < sensor_size[0]):
            x_a[1] += 1
            x_a[0] += 1

        elif vec == 'S' and (x_a[1] + IM_SIZE < sensor_size[1]):
            x_a[1] += 1

        elif vec == 'SW' and (x_a[1] + IM_SIZE < sensor_size[1]) and (x_a[0] - 1 >= 0):
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
            _coord_move(vec=choice(stepset),x_a=x_a,stepset=stepset)
            # print(f"out of frame, step {step}")import itertools

            # coord_move(<- if edge, pick a new )
        
        return x_a
    
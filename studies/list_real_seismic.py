
import os
import numpy as np

# get a list of real seismic volumes
seis_dir = "/Volumes/donaldpg/synthoseis/real_data"
seis_list = os.listdir(seis_dir)
seis_list = [x for x in seis_list if 'npy' in x]
print("seis_list = " + str(seis_list))


# count the number of 128^3 cubes
_n_cube_sum = 0
for i, idata in enumerate(seis_list):
    aaa = np.load(seis_dir + "/" + idata)
    _n_cube = np.array(aaa.shape) // 128
    _n_cube_size = _n_cube.prod()
    _n_cube_sum += _n_cube_size
    print("i, shape = " + str(i) + ", shape = " + str(aaa.shape) + ", subcubes = " + str(_n_cube) + ", " + str(_n_cube_size))
    
print("_n_cube_sum = " + str(_n_cube_sum))

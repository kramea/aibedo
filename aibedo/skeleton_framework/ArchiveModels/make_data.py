import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import h5py
from interpolate import *


data_path = "/Users/sookim/aibedo/skeleton_framework/data/Processed_CESM2_r1i1p1f1_historical_Input.nc"
ds= xr.open_dataset(data_path)
var_list = list(ds.data_vars)
file_all = []
print(var_list)
for var in var_list:
    print(var)
    ours = np.asarray(ds[var][0:10])
    data_all = []
    for i in range(len(ours)):
        da = ours[i]
        lon_list = list(np.asarray(ds[var][0:10].lon))
        lat_list = list(np.asarray(ds[var][0:10].lat))
        lon, lat, interpolated_value = interpolate_SphereIcosahedral(5, da, lon_list, lat_list)
        data = np.asarray([interpolated_value])
        data_all.append(data)
    data_input = np.reshape(np.concatenate(data_all, axis = 0), [-1,10242,1])
    file_all.append(data_input)
data_file=np.concatenate(file_all, 2)
print(np.shape(data_file))
np.save("./data/input.npy", data_file)



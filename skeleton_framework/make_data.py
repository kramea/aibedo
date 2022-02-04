import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import h5py
from interpolate import *


ds=xr.open_dataset("./ours/rsut_Amon_CESM2_historical_r1i1p1f1_gn_185001-201412.nc")
ours = ds.rsut
os.mkdir("./data_ours/")
print(np.shape(ours))
for i in range(len(ours)):
    print(i)
    da = ds.rsut[i]
    lon_list = list(np.asarray(ds.rsut[i].lon))
    lat_list = list(np.asarray(ds.rsut[i].lat))
    lon, lat, interpolated_value = interpolate_SphereIcosahedral(5, da, lon_list, lat_list)
    data = np.asarray([interpolated_value])
    print(np.shape(data))
    np.savez("./data_ours/"+str(i)+".npz", data=data, labels=data)


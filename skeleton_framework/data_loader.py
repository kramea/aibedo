import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import h5py
import xarray as xr
import copy

def load_ncdf(path_to_ncdf):
    ds = xr.open_dataset(path_to_ncdf)
    time, lat, lon = np.shape(ds.rsut) #(1980, 192, 288)
    data = np.reshape(np.asarray(ds.rsut), [time, 1, lat, lon])
    return data

def normalize(data):
    data_ = copy.deepcopy(data)
    n,c,lat,lon = np.shape(data_)
    for i in range(c):
        min_val = np.amin(data[:, i, :,:])
        max_val = np.amax(data[:, i, :,:])
        for j in range(n):
            data_[j,i,:,:] = (data[j,i,:,:]-min_val)/(max_val-min_val)
    return data_



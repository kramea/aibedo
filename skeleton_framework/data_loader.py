import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import h5py
import xarray as xr
from interpolate import *
import copy
import os

def load_ncdf(path_to_ncdf):
    ds = xr.open_dataset(path_to_ncdf)
    time, lat, lon = np.shape(ds.rsut) #(1980, 192, 288)
    data = np.reshape(np.asarray(ds.rsut), [time, 1, lat, lon])
    return data

def normalize(data):
    data_ = copy.deepcopy(data)
    if len(np.shape(data)) == 4:
        n,c,lat,lon = np.shape(data_)
        for i in range(c):
            min_val = np.amin(data[:, i, :,:])
            max_val = np.amax(data[:, i, :,:])
            np.save("./output_sunet/min.npy", min_val)
            np.save("./output_sunet/max.npy", max_val)
            for j in range(n):
                data_[j,i,:,:] = (data[j,i,:,:]-min_val)/(max_val-min_val)
    elif len(np.shape(data)) == 3:
        n, size, c = np.shape(data_)
        for i in range(c):
            min_val = np.amin(data[:, i, :])
            max_val = np.amax(data[:, i, :])
            np.save("./output_sunet/min_"+str(i)+".npy", min_val)
            np.save("./output_sunet/max_"+str(i)+".npy", max_val)
            for j in range(n):
                data_[j,:,i] = (data[j,:,i]-min_val)/(max_val-min_val)
    return data_

def load_ncdf_to_SphereIcosahedral(path_to_ncdf):
    ds=xr.open_dataset(path_to_ncdf)
    ours = ds.rsut
    lon_list = list(np.asarray(ds.rsut[0].lon))
    lat_list = list(np.asarray(ds.rsut[0].lat))
    print(np.shape(ours))
    result = []
    for i in range(500, 1980): #len(ours)):
        print(i)
        da = ds.rsut[i]
        lon, lat, interpolated_value = interpolate_SphereIcosahedral(5, da, lon_list, lat_list)
        n = len(interpolated_value)
        data = np.reshape(interpolated_value, [1,n,1])
        result.append(data)
    output = np.concatenate(result, axis=0)
    return lon, lat, output








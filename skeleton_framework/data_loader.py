import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import h5py
import xarray as xr
from interpolate import *
import copy
import os
import time


def load_ncdf(path_to_ncdf):
    ds = xr.open_dataset(path_to_ncdf)
    time, lat, lon = np.shape(ds.rsut) #(1980, 192, 288)
    data = np.reshape(np.asarray(ds.rsut), [time, 1, lat, lon])
    return data

def normalize(data, parameter):
    data_ = copy.deepcopy(data)
    if len(np.shape(data)) == 4:
        n,c,lat,lon = np.shape(data_)
        min_val = []
        max_val = []
        for i in range(c):
            min_val.append(np.amin(data[:, i, :,:]))
            max_val.append(np.amax(data[:, i, :,:]))
        np.save("./output_sunet/min_"+parameter+".npy", min_val)
        np.save("./output_sunet/max_"+parameter+".npy", max_val)
        for i in range(c):
            for j in range(n):
                data_[j,i,:,:] = (data[j,i,:,:]-min_val[i])/(max_val[i]-min_val[i])
    elif len(np.shape(data)) == 3:
        n, size, c = np.shape(data_)
        min_val = []
        max_val = []
        for i in range(c):
            min_val.append(np.amin(data[:, i, :]))
            max_val.append(np.amax(data[:, i, :]))

        np.save("./output_sunet/min_"+parameter+".npy", min_val)
        np.save("./output_sunet/max_"+parameter+".npy", max_val)
        for i in range(c):
            for j in range(n):
                data_[j,:,i] = (data[j,:,i]-min_val[i])/(max_val[i]-min_val[i])
    return data_




def load_ncdf_to_SphereIcosahedral(data_path):
    #data_path = "/Users/sookim/aibedo/skeleton_framework/data/Processed_CESM2_r1i1p1f1_historical_Input.nc"
    ds= xr.open_dataset(data_path)
    var_list = list(ds.data_vars)
    file_all = []
    print(var_list)
    for var in var_list:
        print(var)
        ours = np.asarray(ds[var])
        data_all = []
        for i in range(len(ours)):
            da = ours[i]
            lon_list = list(np.asarray(ds[var].lon))
            lat_list = list(np.asarray(ds[var].lat))
            start = time.time()
            lon, lat, interpolated_value = interpolate_SphereIcosahedral(6, da, lon_list, lat_list)
            end = time.time()
            print("elapsed time: "+str(end-start)+" secs")
            data = np.asarray([interpolated_value])
            data_all.append(data)
        data_input = np.reshape(np.concatenate(data_all, axis = 0), [-1,len(interpolated_value),1])
        file_all.append(data_input)
    data_file=np.concatenate(file_all, 2)
    return lon, lat, data_file


def load_ncdf(data_path):
    #data_path = "/Users/sookim/aibedo/skeleton_framework/data/Processed_CESM2_r1i1p1f1_historical_Input.nc"
    ds= xr.open_dataset(data_path)
    var_list = list(ds.data_vars)
    file_all = []
    print(var_list)
    for var in var_list:
        ours = np.asarray(ds[var]) #(1980, 192, 288)
        shape = [1]+list(np.shape(ours))
        ours = np.reshape(ours,shape)
        file_all.append(ours)
    data_file = np.concatenate(file_all, 0)
    data_file = np.swapaxes(data_file, 0,1)
    print(np.shape(data_file))
    return data_file

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


"""
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
"""


def load_ncdf_to_SphereIcosahedral(data_path):
    #data_path = "/Users/sookim/aibedo/skeleton_framework/data/Processed_CESM2_r1i1p1f1_historical_Input.nc"
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
    return lon, lat, data_file



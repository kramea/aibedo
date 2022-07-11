import numpy as np
import xarray as xr
import copy, time, random


def load_ncdf(path_to_ncdf):
    ds = xr.open_dataset(path_to_ncdf)
    time, lat, lon = np.shape(ds.rsut)  # (1980, 192, 288)
    data = np.reshape(np.asarray(ds.rsut), [time, 1, lat, lon])
    return data


def shuffle_data(d1, d2):
    n = len(d1)
    m = len(d2)
    assert (n == m)
    idx = [i for i in range(m)]
    random.shuffle(idx)
    d1_out = []
    d2_out = []
    for i in idx:
        d1_out.append(d1[i:i + 1])
        d2_out.append(d2[i:i + 1])
    d1_out2 = np.concatenate(d1_out, axis=0)
    d2_out2 = np.concatenate(d2_out, axis=0)
    return d1_out2, d2_out2


def shuffle_data_meanstd(d1, d2, d3, d4):
    n = len(d1)
    m = len(d2)
    assert (n == m)
    idx = [i for i in range(m)]
    random.shuffle(idx)
    d1_out = []
    d2_out = []
    d3_out = []
    d4_out = []
    for i in idx:
        d1_out.append(d1[i:i + 1])
        d2_out.append(d2[i:i + 1])
        d3_out.append(d3[i:i + 1])
        d4_out.append(d4[i:i + 1])
    d1_out2 = np.concatenate(d1_out, axis=0)
    d2_out2 = np.concatenate(d2_out, axis=0)
    d3_out2 = np.concatenate(d3_out, axis=0)
    d4_out2 = np.concatenate(d4_out, axis=0)
    return d1_out2, d2_out2, d3_out2, d4_out2


def normalize(data, parameter):
    data_ = copy.deepcopy(data)
    if len(np.shape(data)) == 4:
        n, c, lat, lon = np.shape(data_)
        min_val = []
        max_val = []
        for i in range(c):
            min_val.append(np.amin(data[:, i, :, :]))
            max_val.append(np.amax(data[:, i, :, :]))
        np.save("./output_sunet/min_" + parameter + ".npy", min_val)
        np.save("./output_sunet/max_" + parameter + ".npy", max_val)
        for i in range(c):
            for j in range(n):
                data_[j, i, :, :] = (data[j, i, :, :] - min_val[i]) / (max_val[i] - min_val[i])
    elif len(np.shape(data)) == 3:
        n, size, c = np.shape(data_)
        min_val = []
        max_val = []
        for i in range(c):
            min_val.append(np.amin(data[:, :, i]))
            max_val.append(np.amax(data[:, :, i]))

        np.save("./output_sunet/min_" + parameter + ".npy", min_val)
        np.save("./output_sunet/max_" + parameter + ".npy", max_val)
        for i in range(c):
            for j in range(n):
                data_[j, :, i] = (data[j, :, i] - min_val[i]) / (max_val[i] - min_val[i])
        for i in range(c):
            print(np.amin(data_[:, :, i]), np.amax(data_[:, :, i]))
    return data_


def temporal_conversion(data, time):
    """
       data: [T, N, C]
    """
    print("start temporal conversion of original data shaped as " + str(np.shape(data)))
    data = np.swapaxes(data, 1, 2)
    t, _, _ = np.shape(data)
    temporal_data = []
    stride = 1
    for i in range(0, int(t / stride) - time):
        d1, d2, d3 = np.shape(data[i * stride:i * stride + time])
        temporal_data.append(np.reshape(data[i * stride:i * stride + time], [1, d1, d2, d3]))
    out = np.concatenate(temporal_data, axis=0)
    return out


def load_ncdf_to_SphereIcosahedral(data_path, glevel=5, dvar=None):
    from aibedo.skeleton_framework.interpolate import interpolate_SphereIcosahedral
    # data_path = "/Users/sookim/aibedo/skeleton_framework/data/Processed_CESM2_r1i1p1f1_historical_Input.nc"
    ds = xr.open_dataset(data_path)
    print(list(ds.data_vars))
    if dvar == None:
        var_list = [v for v in list(ds.data_vars) if "_pre" in v or "_Pre" in v]
    else:
        var_list = dvar
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
            lon, lat, interpolated_value = interpolate_SphereIcosahedral(glevel, da, lon_list, lat_list)
            end = time.time()
            # print("elapsed time: "+str(end-start)+" secs")
            data = np.asarray([interpolated_value])
            data_all.append(data)
        data_input = np.reshape(np.concatenate(data_all, axis=0), [-1, len(interpolated_value), 1])
        file_all.append(data_input)
    data_file = np.concatenate(file_all, 2)
    return lon, lat, data_file


def load_ncdf(data_path):
    ds = xr.open_dataset(data_path)
    var_list = list(ds.data_vars)
    file_all = []
    print(var_list)
    for var in var_list:
        ours = np.asarray(ds[var])  # (1980, 192, 288)
        shape = [1] + list(np.shape(ours))
        ours = np.reshape(ours, shape)
        file_all.append(ours)
    data_file = np.concatenate(file_all, 0)
    data_file = np.swapaxes(data_file, 0, 1)
    print(np.shape(data_file))
    return data_file

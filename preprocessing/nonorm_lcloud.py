import subprocess as sp
import os
import xarray as xr
import numpy as np
import time

### Adds no-normalized low cloud data to the datasets to avoid issues with divide by zero
### in regions with no low cloud or topography is <700hPa

def removeSC(x:np.ndarray):
    '''
    Removes seasonal cycle from monthly data
    x - 3D (time,lat,lon) numpy array
    '''
    nt,nx = x.shape
    nyears = nt//12
    monmean = np.mean(x.reshape(nyears,12,nx),axis=0)
    for m in range(12):
        x[m::12] = x[m::12] - monmean[m]
    return x

def detrend(x,time):
    '''
    remove degree three polynomial fit
    x - numpy array
    time - 
    '''
    nt,nx = x.shape
    xtemp = x.copy()
    p = np.polyfit(time, xtemp, deg=3)
    print(p.shape)
    fit = p[0]*(time[:,np.newaxis] **3)+ p[1]*(time[:,np.newaxis]**2) + p[2]*(time[:,np.newaxis]) + p[3]
    return x - fit

path = '/home/jupyter-haruki/work/processed/to_s3/'

for fname in os.listdir(path):
    if 'compress' in fname and 'Input.Exp8_fixed' in fname:
        data = xr.open_dataset(path + fname,decode_times=False)
        if 'lcloud_nonorm' in data.variables:
            continue
        print(fname)
        ## remove seasonal cycle
        t1 = time.time()
        print('Deseasonalize')
        deseas = removeSC(data.lcloud.values.copy())
        t2 = time.time()
        print('Time : ', t2-t1)

        print('Detrend')
        ## remove order cubic fit
        print(data.time.values.shape)
        detre = detrend(deseas,data.time.values.copy())
        t3 = time.time()
        print('Time : ',t3-t2)

        ## generate DataArray
        lcloud_nonorm = xr.DataArray(name='lcloud_nonorm', data=detre.astype(np.float32), attrs={"long_name":"Low cloud fraction deseas, detrend, no normalization", 
                                        "units":"%", "Description":"Cloud fraction integrated below 700hPa",
                                        "variables":["cl"],"minlevel":700e2, "maxlevel":1000e2, "integral":"vertical","signs":[1]}, 
                            coords={'time': data.time,'ncells':data.ncells})

        ds = lcloud_nonorm.to_dataset(name = 'lcloud_nonorm')
        del data
        ds.to_netcdf(path=path+fname,mode='a',format='NETCDF4')

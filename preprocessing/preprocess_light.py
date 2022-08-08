import xarray as xr
import numpy as np
from multiprocessing import Pool
import time
import pandas as pd
import json
import os
import sys

### Set of functions for executing preprocessing of CMIP6 data
### In preparation for use by the AiBEDO model
### Light version without S3 loading or vertical interpolation

# Connect to AWS S3 storage
proc = 4

def removeSC(x:np.ndarray):
    '''
    Removes seasonal cycle from monthly data
    x (np.ndarray) - 3D (time,lat,lon) numpy array
    '''
    nt,nx,ny = x.shape # get array dimensions
    nyears = nt//12
    # get month means
    monmean = np.mean(x.reshape(nyears,12,nx,ny),axis=0)
    for m in range(12): #for each month
        x[m::12] = x[m::12] - monmean[m]
    return x

def detrend(x:np.ndarray,time:np.ndarray):
    '''
    remove degree three polynomial fit
    x (np.ndarray) : 3D (time, lat, lon) numpy array
    time (np.ndarray) : 1D (time) array 
    '''
    nt,nx,ny = x.shape
    xtemp = x.reshape(nt,nx*ny)
    p = np.polyfit(time, xtemp, deg=3)
    print(p.shape)
    fit = p[0]*(time[:,np.newaxis] **3)+ p[1]*(time[:,np.newaxis]**2) + p[2]*(time[:,np.newaxis]) + p[3]
    return x - fit.reshape(nt,nx,ny)

def preprocess(data:np.ndarray, tdata:np.ndarray, window:int = 31,proc:int = 40):
    '''
    Executes the three steps of the preprocessing (deseasonalize, detrend, normalize)
    data (np.ndarray) : data to process
    t (np.ndarray) : time values
    window (integer) : years for the rolling average
    proc (int) : number of processes for multiprocessing
    '''

    ### remove seasonal cycle
    t1 = time.time()
    print('Deseasonalize')
    deseas = removeSC(data)
    t2 = time.time()
    print('Time : ', t2-t1)

    ### remove cubic fit
    print('Detrend')
    print(tdata.shape)
    print(deseas.shape)
    detre = detrend(deseas,tdata)
    t3 = time.time()
    print('Time : ',t3-t2)

    x = detre

    ## function for normalizing - written for passing to multiprocessing
    global normalize
    def normalize(t):
        '''
        Calculate anomaly and normalize (repeat boundary values) for monthly data
        averages across years (12 time step skips)
        x (np.ndarray) - integer values
        t (integer) - time
        window (integer) - years in the averaging window must be odd for now
        '''
        assert window%2 == 1
        tmax = x.shape[0]
        halfwindow = window//2
        yr = t//12
        mon = t%12
        selx = np.zeros_like(x[:window])
        # get rolling window (backfills/forward fills with first/last value)
        if t-halfwindow*12 < 0:
            selx[:halfwindow - yr] = x[mon]
            selx[halfwindow-yr:] = x[mon:t+halfwindow*12+1:12]
        elif t+halfwindow*12+1 > tmax:
            selx[halfwindow + (tmax//12 - yr):] = x[tmax- 12 + mon]
            selx[:halfwindow + (tmax//12 - yr)] = x[t - halfwindow*12::12]
        else:
            selx = x[t-halfwindow*12:t+halfwindow*12+1:12]
        # calculate normalized
        normed = (x[t] - np.mean(selx,axis=0))/np.std(selx,axis=0)
        return normed
    ### run normalization
    print('Normalize')
    ntime = x.shape[0]

    # parallelize normalizing across time (trivially parallel)
    with Pool(processes=proc) as pool: 
        outdata = pool.map(normalize, range(ntime))
    t4 = time.time()
    print('Time : ',t4-t3)
    return np.array(outdata,dtype=np.float32)

def calc_var(data:dict, variables:list, signs:list, varname:str, attributes:dict = {},years:tuple = None):
    '''
    Handles calculation and preprocessing of variables
    written for calculating different radiation terms where a sequence of radiation variables are differenced
    data (dict) : dictionary of xr.Dataset with the input CMIP variables
    variables (list) : list of input CMIP variables used to calculate the output variable
    signs (list) : list of weights/signs used to calculate the output variable 
        (e.g. toa cloud shortwave crel = fsnt-fsntcs would have variables = [fsnt, fsntcs] and signs=[1,-1])
    varname (str) : output variable name
    attributes (dict) : dictionary with attributes to give to the xr.DataArray() defining variable information
        such as units, etc
    '''
    # using weights from the attribute json file
    # calculate the values to process
    # (written with calculating radiation variables in mind)
    if not years is None:
        nvar = len(variables)
        var1 = variables[0]
        val = signs[0]*data[var1][var1].sel(time = slice(365*years[0],365*(years[1]+1))).values # note we need to convert to np.ndarray to add the arrays together
        for i in range(1,nvar):
            var = variables[i]
            val += signs[i]*data[var][var].sel(time = slice(365*years[0],365*(years[1]+1))).values
        val = val.astype(np.float32) # convert to np.float32 to reduce storage size when writing to netcdf
    else:
        nvar = len(variables)
        var1 = variables[0]
        val = signs[0]*data[var1][var1].values # note we need to convert to np.ndarray to add the arrays together
        for i in range(1,nvar):
            var = variables[i]
            val += signs[i]*data[var][var].values
        val = val.astype(np.float32) # convert to np.float32 to reduce storage size when writing to netcdf

    # Check to see if the values have been properly calculated
    if np.max(val[0]) == 0 and np.min(val[0]) == 0:
        raise Exception(varname+' calculation went wrong')
    if np.any(np.isnan(val)):
        raise Exception(varname + ' calculation went wrong')

    # Run the preprocessing
    # note we need to copy the data and time arrays to avoid changing the values in the input dataset
    print(val.shape)
    out = preprocess(np.squeeze(val.copy()),data[var1].time.values.copy())

    # convert output np.ndarrays to xarray DataArrays
    attributes_processed = attributes.copy()
    if 'long_name' in attributes_processed: 
        attributes_processed['long_name'] += ' Pre-processed'
    processed = xr.DataArray(name=varname + '_pre', data=out, attrs=attributes, 
                                coords={'time': data[var1].time,'lat': data[var1].lat,'lon': data[var1].lon})
    orig= xr.DataArray(name=varname, data=np.squeeze(val), attrs=attributes, 
                                coords={'time': data[var1].time,'lat': data[var1].lat,'lon': data[var1].lon})
    return processed, orig

def run_preprocess(experiment:str, modelName:str, member:str, variables_in:list,
            variables_out:list, attribute_path:str = 'variable_defs.json',
            out_path:str = None,lf_path:str = None,data_path:str = None,
            append:bool=False,frequency:str='Amon',
            years:tuple=None):
    '''
    Wrapper function that executes the preprocessing pipeline
    experiment (str) : experiment name (experiment_id)
    modelName (str) : model name (source_id)
    member (str) : ensemble member code (member_id) r*i*p*f*
    variables_in (list) : list of variables to grab from CMIP6 archive
    variables_out (list) : list of variables to calculate and output
    attribute_path (str) : path to a json file with variable descriptions.
        Note if you request a variable in variables_out that is not a CMOR variable, it must
        be defined in the attribute .json
    data_path (str): path to CMIP6 data
    out_path (str) : file name for output
    lf_path (str) : path to sftlf data
    append (str) : append variables_out to an existing netcdf file
    frequency (str) : CMIP table to get data from (e.g. Amon, Omon, Aday - table_id). 
    '''
    # Default out_path
    if out_path is None: 
        out_path = '/home/jupyter-haruki/work/{0}_{1}_{2}_output.nc'.format(modelName, member, experiment)
    if not os.path.isdir(data_path):
        raise Exception("Invalid path to data")

    print('Preprocessing for {0} {1} {2}'.format(modelName,experiment,member))

    # if we are appending to an existing netcdf
    # load the dataset and determine the existing variables
    if append:
        if os.path.isfile(out_path):
            existing = xr.open_dataset(out_path)
            existing_variables = list(existing.variables)
            del existing
        else:
            print('append = True, but no existing file. Setting to append = False')
            append = False

    ## Load in data (could make faster with open_mfdataset?)
    #dict_query = {'source_id':modelName, 'table_id':frequency, 'experiment_id':experiment, 'member_id':member}
    data = {}
    for var in variables_in:
        filename = '{0}.{1}.{2}.{3}.{4}.nc'.format(modelName,experiment,member,frequency,var)
        data[var] = xr.open_dataset(data_path + filename, decode_times=False)

    # Load land fraction data
    if lf_path: # set lf_path=False to skip
        data['sftlf'] = xr.open_dataset(lf_path) ## land area fraction as %

    # Attributes for derived variables
    d_attr = json.load(open(attribute_path))

    processed = {} # preprocessed data
    orig = {} # raw data
    for var in variables_out:
        if append and var in existing_variables:
            # skip existing variables if in append mode
            print('{0} already exists, skipping'.format(var))
            variables_out.remove(var)
            continue
        print('-----------------')
        print(var)
        print('-----------------')
        if var in variables_in: 
            # If taking variable directly from CMIP
            signs = [1]
            variables = [var]
            attributes = {'long_name': data[var][var].long_name, 'units':data[var][var].units}
        elif var in d_attr: 
            # If calculating a variable from CMIP variables
            signs = d_attr[var]['signs']
            attributes = d_attr[var]
            variables = d_attr[var]['variables']
        else: # Something's wrong (missing variable or definition)
            print('missing variable definition in {0}'.format(attribute_path))
            continue
        if 'integral' in attributes:
            # if we are integrating a 3D variable
            #processed[var], orig[var] = calc_var_integral(data,variables[0],var,attributes = attributes)
            raise Exception("Run full preprocess_parallel for vertical interpolation")
        else:
            processed[var], orig[var] = calc_var(data, variables, signs, var,attributes = attributes,years=years)
    
    # Cast the fixed land fraction data into a time series
    if lf_path: # set lf_path=False to skip
        # tile in time
        sftlfOut, b2 = xr.broadcast(data['sftlf'].sftlf, data[variables_in[0]][variables_in[0]], exclude=('lat','lon'))
        sftlfOut=sftlfOut.where(sftlfOut==0,1) ### data has values as 0 for ocean or 100 for land sa making 100 as 1
        lsMask = xr.DataArray(name='lsMask', data=sftlfOut.load(), attrs={'long_name':'land_sea_mask', 'Description':'land fraction of each grid cell 0 for ocean and 1 for land'}, 
                            coords={'time': data[variables_in[0]].time,'lat': data['sftlf'].lat,'lon': data['sftlf'].lon})

    # change time to original values (time values get altered in calc_var)
    for var in variables_out:
        processed[var]['time'] = data[variables_in[0]].time
        orig[var]['time'] = data[variables_in[0]].time

    # Save one DataArray as dataset
    var = variables_out[0]
    output_ds = processed[var].to_dataset(name = var+'_pre')
    output_ds[var] = orig[var]
    for var in variables_out[1:]:
        # Add next DataArray to existing dataset (ds)
        output_ds[var+'_pre'] = processed[var]
        output_ds[var] = orig[var]
    # add the landfraction query
    if not lf_path is None:
        output_ds['lsMask'] = lsMask
    print('Write to file')
    if append:
        output_ds.to_netcdf(path=out_path,mode='a',format='NETCDF4')
    else:
        output_ds.to_netcdf(path=out_path,mode='w',format='NETCDF4')

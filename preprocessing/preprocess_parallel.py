import xarray as xr
import numpy as np
from multiprocessing import Pool
import s3fs
import geocat.comp
import time
import pandas as pd
import json
import os
import sys
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)
sys.path.insert(0,'/home/jupyter-haruki/preprocessing/vinth2p/src')
# Import Deepak Chandan's vinth2p_ecmwf implementation
import interp

### Set of functions for executing preprocessing of CMIP6 data
### In preparation for use by the AiBEDO model

# Connect to AWS S3 storage
fs = s3fs.S3FileSystem(anon=True)
df = pd.read_csv("https://cmip6-pds.s3.amazonaws.com/pangeo-cmip6.csv")

proc = 40

def getData(query:dict):
    '''
    Load AWS CMIP6 data into xarray dataframe
    query (dict or str) - dict or str with data information
                        - if dict format as {'param':'value','param2':['val1','val2']}
    '''
    # Create query string for pandas.DataFrame.query
    if type(query) is dict:
        inputStr = " & ".join(["{0}=='{1}'".format(param, query[param]) for param in query])
    elif type(query) is str: # if its already a string, pass through
        inputStr=query

    # Searches cmip6 data csv for datasets that match given parameters
    df_subset = df.query(inputStr)
    if df_subset.empty:
        print('data not available for '+inputStr)
    else:
        # load data
        for v in df_subset.zstore.values:
            zstore = v
            mapper = fs.get_mapper(zstore)
            ### !!!! Note decode times is false so we can use integer time values !!!!
            ### open_zarr, so datasets are not loaded yet
            return_ds = xr.open_zarr(mapper, consolidated=True)
    return(return_ds)

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

def vert_int(data:xr.Dataset, a:np.ndarray, b:np.ndarray, ps:np.ndarray,
            p_max:float, p_min:float, plevels:np.ndarray = None, usegeocat:bool = False):
    '''
    Calculate the vertical integral between two levels. Assuming data is in model coordinates,
    we first remap the data to pressure levels using vinth2p/geocat
    data (xr.Dataset) : 4D input dataset in hybrid pressure-sigma coordinates
    a (np.ndarray) : hybrid pressure-sigma "a" values (0 to 1)
    b (np.ndarray) : hybrid pressure-sigma "b" values (0 to 1)
    ps (np.ndarray) : 3D input dataset with surface pressures (Pa)
    p_max (float) : maximum pressure level (Pa)
    p_min (float) : minimum pressure level (p_min < p_max) (Pa)
    plevels (np.ndarray) : array of target pressure levels for interpolation (Pa)
    usegeocat (bool) : switch to geocat interpolation
    '''
    if plevels is None:
        # default levels in Pa
        default_levels = np.array([100000., 92500., 85000., 70000., 50000., 40000., 30000., 25000., 
                                20000., 15000., 10000., 7000., 5000., 3000., 2000., 1000., 700.,
                                500., 300., 200., 100.],dtype=np.float32)
        if not usegeocat:
            # formatting for vinth2p
            # reverse order of pressures (ascending, in hPa)
            ps = ps/1e2 # convert to hPa
            p_max = p_max/1e2 # convert to hPa
            p_min = p_min/1e2 # convert to hPa
            P0 = 1000.
            # change to ascending pressures if the pressure is descending
            if b[1] - b[0] < 0:
                default_levels = default_levels[::-1]/1e2
                b = b[::-1].values
                a = a[::-1].values
                data = data[:,::-1]
            else:
                default_levels = default_levels/1e2
                b = b.values
                a = a.values
        plevels = default_levels[(default_levels >= p_min) & (default_levels <= p_max)]
    if plevels.dtype != 'float32':
        plevels = plevels.astype(np.float32)
    print(plevels)

    ## define function to run vinth2p for multiprocesing map
    global worker
    def worker(t):
        print(t)
        # plevels,a,b,lev must be ascending (from TOA to Surf)
        # ps, P0 must be hPa
        temp = ps[t].values
        outdata = interp.vinth2p_ecmwf_fast(data[t].values, a, b, P0,ps[t].values, plevels, temp, temp, 3, 1,1,0)
        return outdata

    ### run vertical interpolation
    if usegeocat:
        # use geocat
        data_interp = geocat.comp.interpolation.interp_hybrid_to_pressure(data,ps,a,b,
                                    new_levels=plevels)
    else:
        # vinth2p
        ntime = data.shape[0]
        with Pool(processes=proc) as pool:
            outdata = pool.map(worker, range(ntime))
        data_interp = xr.DataArray(data = np.array(outdata),dims=['time','plev','lat','lon'],
                                    coords={'lon':(['lon'],data.lon.values),'lat':(['lat'],data.lat.values),
                                            'time':(['time'],data.time.values),'plev':(['plev'],plevels*1e2),})
        ps = ps*1e2 # convert to Pa
        p_max = p_max*1e2
        p_min = p_min*1e2
        data_interp = data_interp.where((data_interp.plev < ps).transpose("time","plev","lat","lon")).fillna(0.)

    integral = vert_int_press(data_interp, ps, p_max, p_min)
    return integral
    '''
    # Slice up levels
    if plevels[0] < plevels[1]:
        data_sel = data_interp.sel(plev=slice(p_min,p_max))
    elif plevels[0] > plevels[1]:
        data_sel = data_interp.sel(plev=slice(p_max,p_min))

    ### calculate vertical integral
    pressure_levels = data_sel.plev
    pressure_levels=pressure_levels.assign_attrs({'units':'Pa'})

    # use geocat to get the weights for pressure interpolation
    var_weights=data_sel.copy()
    var_weights.values=geocat.comp.dpres_plevel(pressure_levels,ps,p_min).fillna(0.0).values

    # calculate weighted mean
    integral=data_sel.weighted(var_weights).mean(dim='plev')
    return integral
    '''

def vert_int_press(data:xr.Dataset, ps:np.ndarray, p_max:float, p_min:float):
    '''
    Calculate the vertical integral between two levels for data on pressure coordinates
    data (xr.Dataset) : 4D input dataset in pressure coordinates
    ps (np.ndarray) : 3D input dataset with surface pressures (Pa)
    p_max (float) : maximum pressure level (Pa)
    p_min (float) : minimum pressure level (p_min < p_max) (Pa)
    plevels (np.ndarray) : array of target pressure levels for interpolation (Pa)
    '''

    # Slice up levels
    if data.plev[0] < data.plev[1]:
        data_sel = data.sel(plev=slice(p_min,p_max))
    elif data.plev[0] > data.plev[1]:
        data_sel = data.sel(plev=slice(p_max,p_min))

    ### calculate vertical integral
    pressure_levels = data_sel.plev
    pressure_levels=pressure_levels.assign_attrs({'units':'Pa'})

    # use geocat to get the weights for pressure interpolation
    var_weights=data_sel.copy()
    var_weights.values=geocat.comp.dpres_plevel(pressure_levels,ps,p_min).fillna(0.0).values

    # calculate weighted mean
    integral=data_sel.weighted(var_weights).mean(dim='plev')

    return integral
    
def calc_var_integral(data:dict, var:str, varname:str, attributes:dict = {},
             P0:float=1000e2, usegeocat:bool=False):
    '''
    Runs processing for a variable including a vertical integration step (namely low cloud)
    Can either use geocat or a cython implementation of vinth2p
    data (dict) : dictionary of xr.Dataset()'s of input variables from CMIP
    var (str) : 3D CMIP variable to integrate
    varname (str) : output variable name
    attribute (dict) : attributes to give to xr.DataArray()
    usegeocat (bool) : switch to using geocat
    '''
    # get the max and min pressure levels for the integration
    if 'maxlevel' in attributes: 
        p_max = attributes['maxlevel']
    else: 
        p_max = 1000e2
        attributes['maxlevel'] = p_max
    if 'minlevel' in attributes: 
        p_min = attributes['minlevel']
    else: 
        p_min = 200e2
        attributes['minlevel'] = p_min

    # get surface pressure data
    if 'ps' in data[var].variables:
        ps = data[var].ps.load()
    elif 'ps' in data:
        ps = data['ps']['ps'].load()
        print(ps.shape)
    else:
        raise Exception('No surface pressure data')
        
    if 'lev' in data[var].variables:
        ### parse different hybrid pressure-sigma coordinate weight formats
        if 'formula' in data[var]['lev'].attrs:
            # if there is a formula provided, we can use it to standardize the a,b arrays
            formula = data[var]['lev'].attrs['formula']
            if formula == 'p = ptop + sigma*(ps - ptop)':
                # FGOALS
                sigma = data[var]['lev'].load()
                ptop = data[var]['ptop'].load()
                hy_a = (1-sigma) * ptop/P0
                hy_b = sigma
            elif formula == 'p(n,k,j,i) = ap(k) + b(k)*ps(n,j,i)' or formula == 'p = ap + b*ps':
                #GFDL, AWI
                hy_a = data[var]['ap'].load()/P0
                hy_b = data[var]['b'].load()
            elif formula == 'p = a*p0 + b*ps':
                #most other hybrid sigma-press
                hy_a = data[var]['a'].load()
                hy_b = data[var]['b'].load()
                if data[var]['p0'].values == 1: # special case for GISS-E2-1-H
                    hy_a = data[var]['a'].load()/1000e2
            elif formula == 'z = a + b*orog':
                # UK models use hybrid sigma-level, implement later if needed
                raise Exception('Can\'t do hybrid sigma-meters at this time')
        else:
            if 'a' in data[var].variables and 'b' in data[var].variables:
                hy_a = data[var]['a'].load()
                hy_b = data[var]['b'].load()
            else:
                raise Exception('dont understand vertical coordinate')

        ds = data[var][var].load()

        ## compute integral
        integrated = vert_int(ds.copy(), hy_a, hy_b, ps, p_max, p_min, usegeocat=usegeocat)
    elif 'plev' in data[var].variables:
        ds = data[var][var].load()
        integrated=vert_int_press(ds, ps, p_max, p_min)
    else:
        raise Exception;
    ## convert to array for the vertical integrated data
    print(integrated)
    orig= xr.DataArray(name=varname, data=integrated, attrs=attributes, 
                                coords={'time': data[var].time,'lat': data[var].lat,'lon': data[var].lon})
    if np.max(integrated[0]) == 0 and np.min(integrated[0]) == 0:
        raise Exception(varname+' calculation went wrong')

    ## run the preprocessing
    time_array = np.array(data[var].indexes['time'].to_datetimeindex(),dtype=float)/1e9/60/60/24
    out = preprocess(integrated.copy().values,time_array)
    attributes_processed = attributes.copy()
    if 'long_name' in attributes_processed: 
        attributes_processed['long_name'] += ' Pre-processed'
        
    ## convert to array for the vertical integrated data
    processed = xr.DataArray(name=varname + '_pre', data=out, attrs=attributes, 
                                coords={'time': data[var].time,'lat': data[var].lat,'lon': data[var].lon})
    return processed, orig

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
    # We have time in cftime.DatetimeNoLeap values, so we convert to datetime64[ns] in float values then
    # convert to days
    time_array = np.array(data[var1].indexes['time'].to_datetimeindex(),dtype=float)/1e9/60/60/24
    out = preprocess(val.copy(),time_array)

    # convert output np.ndarrays to xarray DataArrays
    attributes_processed = attributes.copy()
    if 'long_name' in attributes_processed: 
        attributes_processed['long_name'] += ' Pre-processed'
    processed = xr.DataArray(name=varname + '_pre', data=out, attrs=attributes, 
                                coords={'time': data[var1].time,'lat': data[var1].lat,'lon': data[var1].lon})
    orig= xr.DataArray(name=varname, data=val, attrs=attributes, 
                                coords={'time': data[var1].time,'lat': data[var1].lat,'lon': data[var1].lon})
    return processed, orig


def run_preprocess(experiment:str, modelName:str, member:str, variables_in:list,
            variables_out:list, attribute_path:str = 'variable_defs.json',
            out_path:str = None,lf_query:str = None,
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
    out_path (str) : file name for output
    lf_query (str) : query string to pass to pandas.DataFram.query  recommended to pass this,
        otherwise the script is likely to fail due to not finding sftlf variable
        set to False to skip adding the landfraction, set to None to attempt to find land fraction
        for the selected experiment and ensemble member
    append (str) : append variables_out to an existing netcdf file
    frequency (str) : CMIP table to get data from (e.g. Amon, Omon, Aday - table_id). 
    '''
    # Default out_path
    if out_path is None: 
        out_path = '/home/jupyter-haruki/work/{0}_{1}_{2}_output.nc'.format(modelName, member, experiment)

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
    dict_query = {'source_id':modelName, 'table_id':frequency, 'experiment_id':experiment, 'member_id':member}
    data = {}
    for var in variables_in:
        dict_query['variable_id'] = var
        data[var] = getData(dict_query)
    ## select years
    if years:
        for var in data:
            #print(data[var].time.max,data[var].time.min)
            data[var] = data[var].sel(time = slice('{0}-01-01'.format(years[0]), '{0}-12-31'.format(years[1])))
        
    # Load land fraction data
    if lf_query is None:
        # if query isn't provided, attempt to load from current experiment
        lf_query  = "source_id=='{0}' & table_id=='fx' & experiment_id=='{1}' &  member_id=='{2}' & variable_id=='sftlf'".format(modelName,experiment,member)
    elif lf_query: # set lf_query=False to skip
        data['sftlf'] = getData(lf_query) ## land area fraction as %

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
            processed[var], orig[var] = calc_var_integral(data,variables[0],var,attributes = attributes)
        else:
            processed[var], orig[var] = calc_var(data, variables, signs, var,attributes = attributes,years=years)
    
    # Cast the fixed land fraction data into a time series
    if lf_query: # set lf_query=False to skip
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
    if lf_query:
        output_ds['lsMask'] = lsMask
    print('Write to file')
    if append:
        output_ds.to_netcdf(path=out_path,mode='a',format='NETCDF4')
    else:
        output_ds.to_netcdf(path=out_path,mode='w',format='NETCDF4')

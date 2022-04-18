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
            return_ds = xr.open_zarr(mapper, consolidated=True,decode_times=False)
    return(return_ds)

def removeSC(x:np.ndarray):
    '''
    Removes seasonal cycle from monthly data
    x - 3D (time,lat,lon) numpy array
    '''
    nt,nx,ny = x.shape # get array dimensions
    nyears = nt//12
    # get month means
    monmean = np.mean(x.reshape(nyears,12,nx,ny),axis=0)
    for m in range(12): #for each month
        x[m::12] = x[m::12] - monmean[m]
    return x

def detrend(x,time):
    '''
    remove degree three polynomial fit
    x - numpy array
    time - 
    '''
    nt,nx,ny = x.shape
    xtemp = x.reshape(nt,nx*ny)
    p = np.polyfit(time, xtemp, deg=3)
    print(p.shape)
    fit = p[0]*(time[:,np.newaxis] **3)+ p[1]*(time[:,np.newaxis]**2) + p[2]*(time[:,np.newaxis]) + p[3]
    return x - fit.reshape(nt,nx,ny)

def preprocess(data,tdata,window=31):
    '''
    data (np.ndarray) : data to process
    t (np.ndarray) : time values
    window (integer) : years for the rolling average
    '''
    t1 = time.time()
    print('Deseasonalize')
    deseas = removeSC(data)
    t2 = time.time()
    print('Time : ', t2-t1)

    print('Detrend')
    print(tdata.shape)
    detre = detrend(deseas,tdata)
    t3 = time.time()
    print('Time : ',t3-t2)

    x = detre

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
        if t-halfwindow*12 < 0:
            selx[:halfwindow - yr] = x[mon]
            selx[halfwindow-yr:] = x[mon:t+halfwindow*12+1:12]
        elif t+halfwindow*12+1 > tmax:
            selx[halfwindow + (tmax//12 - yr):] = x[tmax- 12 + mon]
            selx[:halfwindow + (tmax//12 - yr)] = x[t - halfwindow*12::12]
        else:
            selx = x[t-halfwindow*12:t+halfwindow*12+1:12]
        normed = (x[t] - np.mean(selx,axis=0))/np.std(selx,axis=0)
        return normed
    print('Normalize')
    ntime = x.shape[0]
    with Pool(processes=proc) as pool:
        outdata = pool.map(normalize, range(ntime))
    t4 = time.time()
    print('Time : ',t4-t3)
    return np.array(outdata,dtype=np.float32)

def vert_int(data, a, b, ps,p_max,p_min,plevels = None,timechunk=12, usegeocat = False):
    if plevels is None:
        default_levels = np.array([100000., 92500., 85000., 70000., 50000., 40000., 30000., 25000., 
                                20000., 15000., 10000., 7000., 5000., 3000., 2000., 1000., 700.,
                                500., 300., 200., 100.],dtype=np.float32)
        if not usegeocat:
            # formatting for vinth2p
            # reverse order of pressures (ascending, in hPa)
            ps = ps/1e2 # convert to hPa
            p_max = p_max/1e2
            p_min = p_min/1e2
            P0 = 1000.
            # change to ascending pressures
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
    global worker
    def worker(t):
        print(t)
        # plevels,a,b,lev must be ascending (from TOA to Surf)
        # ps, P0 must be hPa
        temp = ps[t].values
        outdata = interp.vinth2p_ecmwf_fast(data[t].values, a, b, P0,ps[t].values, plevels, temp, temp, 3, 1,1,0)
        return outdata

    if usegeocat:
        data_interp = geocat.comp.interpolation.interp_hybrid_to_pressure(data,ps,a,b,
                                    new_levels=plevels)
    else:
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
    ### get weights depending on thickness of level
    if plevels[0] < plevels[1]:
        data_sel = data_interp.sel(plev=slice(p_min,p_max))
    elif plevels[0] > plevels[1]:
        data_sel = data_interp.sel(plev=slice(p_max,p_min))

    pressure_levels = data_sel.plev
    pressure_levels=pressure_levels.assign_attrs({'units':'Pa'})
    var_weights=data_sel.copy()
    var_weights.values=geocat.comp.dpres_plevel(pressure_levels,ps,p_min).fillna(0.0).values
    ### calculate weighted mean
    integral=data_sel.weighted(var_weights).mean(dim='plev')
    return integral
    
def calc_var_integral(data, var, varname, attributes = {}, P0=1000e2,timechunk=20):
    '''
    Runs processing for a variable we want to integrate
    namely low cloud
    '''
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

    ## attempt to determine vertical coordinate format
    #print(data[var]['lev'])
    if 'formula' in data[var]['lev'].attrs:
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

    # get surface pressure data
    if 'ps' in data[var].variables:
        ps = data[var].ps.load()
    elif 'ps' in data:
        ps = data['ps']['ps'].load()
        print(ps.shape)
    else:
        raise Exception('No surface pressure data')
    ds = data[var][var].load()
    # compute integral
    integrated = vert_int(ds, hy_a, hy_b, ps, p_max, p_min, usegeocat=False)
    print(integrated)
    orig= xr.DataArray(name=varname, data=integrated, attrs=attributes, 
                                coords={'time': data[var].time,'lat': data[var].lat,'lon': data[var].lon})
    if np.max(integrated[0]) == 0 and np.min(integrated[0]) == 0:
        raise Exception(varname+' calculation went wrong')

    out = preprocess(integrated.copy().values,data[var].time.values.copy())
    attributes_processed = attributes.copy()
    if 'long_name' in attributes_processed: 
        attributes_processed['long_name'] += ' Pre-processed'
    processed = xr.DataArray(name=varname + '_pre', data=out, attrs=attributes, 
                                coords={'time': data[var].time,'lat': data[var].lat,'lon': data[var].lon})
    return processed, orig

def calc_var(data, variables, signs, varname,attributes = {}):
    '''
    Handles calculation and preprocessing of variables
    written for calculating different radiation terms where a sequence of radiation variables are differenced
    '''
    nvar = len(variables)
    var1 = variables[0]
    val = signs[0]*data[var1][var1].values
    for i in range(1,nvar):
        var = variables[i]
        val += signs[i]*data[var][var].values
    val = val.astype(np.float32)
    print(np.mean(val[0]))
    if np.max(val[0]) == 0 and np.min(val[0]) == 0:
        raise Exception(varname+' calculation went wrong')
    if np.any(np.isnan(val)):
        raise Exception(varname + ' calculation went wrong')
    out = preprocess(val.copy(),data[var1].time.values.copy())
    attributes_processed = attributes.copy()
    if 'long_name' in attributes_processed: 
        attributes_processed['long_name'] += ' Pre-processed'
    processed = xr.DataArray(name=varname + '_pre', data=out, attrs=attributes, 
                                coords={'time': data[var1].time,'lat': data[var1].lat,'lon': data[var1].lon})
    orig= xr.DataArray(name=varname, data=val, attrs=attributes, 
                                coords={'time': data[var1].time,'lat': data[var1].lat,'lon': data[var1].lon})
    return processed, orig


def run_preprocess(experiment, modelName,member,variables_in,variables_out,
            attribute_path = 'variable_defs.json',out_path = '',lf_query = None,
            append=False,frequency='Amon'):
    
    if out_path == '':
        out_path = '/home/jupyter-haruki/work/{0}_{1}_{2}_output.nc'.format(modelName, member, experiment)

    if append:
        if os.path.isfile(out_path):
            existing = xr.open_dataset(out_path)
            existing_variables = list(existing.variables)
            del existing
        else:
            print('append = True, but no existing file. Setting to append = False')
            append = False

    ## Load in data
    dict_query = {'source_id':modelName, 'table_id':frequency, 'experiment_id':experiment, 'member_id':member}
    data = {}
    for var in variables_in:
        dict_query['variable_id'] = var
        data[var] = getData(dict_query)
    #inputStr  = "source_id=='{0}' & table_id=='Amon' & experiment_id=='{1}' &  member_id=='{2}' & variable_id==".format(modelName,experiment,member)
    #data = {var:getData(inputStr+"'{0}'".format(var)) for var in variables_in}

    print(experiment,member)

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
            print('{0} already exists, skipping'.format(var))
            variables_out.remove(var)
            continue
        print('-----------------')
        print(var)
        print('-----------------')
        if var in variables_in: # If taking variable directly from CMIP
            signs = [1]
            variables = [var]
            attributes = {'long_name': data[var][var].long_name, 'units':data[var][var].units}
        elif var in d_attr: # If calculating a variable from CMIP variables
            signs = d_attr[var]['signs']
            attributes = d_attr[var]
            variables = d_attr[var]['variables']
        else: # Something's wrong (missing variable or definition)
            print('missing variable definition in {0}'.format(attribute_path))
            continue
        if 'integral' in attributes:
            processed[var], orig[var] = calc_var_integral(data,variables[0],var,attributes = attributes)
        else:
            processed[var], orig[var] = calc_var(data, variables, signs, var,attributes = attributes)
    
    if lf_query: # set lf_query=False to skip
        # tile in time
        sftlfOut, b2 = xr.broadcast(data['sftlf'].sftlf, data[variables_in[0]][variables_in[0]], exclude=('lat','lon'))
        sftlfOut=sftlfOut.where(sftlfOut==0,1) ### data has values as 0 for ocean or 100 for land sa making 100 as 1
        lsMask = xr.DataArray(name='lsMask', data=sftlfOut.load(), attrs={'long_name':'land_sea_mask', 'Description':'land fraction of each grid cell 0 for ocean and 1 for land'}, 
                            coords={'time': data[variables_in[0]].time,'lat': data['sftlf'].lat,'lon': data['sftlf'].lon})

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
    if lf_query:
        output_ds['lsMask'] = lsMask
    print('Write to file')
    if append:
        output_ds.to_netcdf(path=out_path,mode='a',format='NETCDF4')
    else:
        output_ds.to_netcdf(path=out_path,mode='w',format='NETCDF4')

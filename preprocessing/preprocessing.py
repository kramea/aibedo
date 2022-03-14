## A set of functions to perform the preprocessing for the AIBEDO model
## using as input xarray datasets

# Import Libraries
import xarray as xr
import pandas as pd
import numpy as np
import iris
import s3fs
import geocat.comp
from datetime import datetime
import dask
from multiprocessing.pool import ThreadPool 
import time

def getData(qstring):
    df_subset = df.query(qstring)
    if df_subset.empty:
        print('data not available for '+qstring)
    else:
        for v in df_subset.zstore.values:
            zstore = v
            mapper = fs.get_mapper(zstore)
            return_ds = xr.open_zarr(mapper, consolidated=True)
    return(return_ds)

def remove_time_mean(x):
    return x - x.mean(dim='time')

def removeSC(x):
    '''
    Remove seasonal cycle
    '''
    return x.groupby('time.month').apply(remove_time_mean)
    
def detrend(x, dName):
    '''
    remove degree three polynomial fit
    x - xr.DataFrame
    dName - string (dimension name)
    '''
    p = x.polyfit(dim=dName, deg=3)
    fit = xr.polyval(coord=x[dName], coeffs=p.polyfit_coefficients)
    return x - fit 

def calStdNorAnom(x):
    '''
    Calculate anomaly and normalize
    x - xr.DataFrame
    '''
    a=[]
    # for each month
    for m in np.unique(x.time.dt.month):
        # select data for the given month
        mData=x[x.time.dt.month==m]
        # calculate the rolling mean and standard deviation and back, forward fill nan values at start, end of time series
        mRolling=mData.rolling(time=31, center=True).mean().bfill(dim="time").ffill(dim="time")
        sRolling=mData.rolling(time=31, center=True).std().bfill(dim="time").ffill(dim="time")
        # calculate normalized values
        normData=(mData-mRolling)/sRolling
        a.append(normData)
    # convert to xarray DataArray
    combineArray=xr.concat(a,'time')
    outArray=combineArray.sortby('time')
    return outArray

def preprocess(var):
    '''
    Performs averaging, normalization
    var - xr.DataFrame with data for a given variable
    '''
    ### remove Seasonal Cycle
    var_RSC=removeSC(var)
    ### Detrend
    var_RSC_dt=detrend(var_RSC,'time')
    ## Calculate Standard normal anomaly 
    var_FinalOut=calStdNorAnom(var_RSC_dt)
    return var_FinalOut

def getLowCloud(var_ds, ds_ps = None):
    ### convert hybrid sigma level to pressure levels
    #var_pr=geocat.comp.interpolation.interp_hybrid_to_pressure(var_ds.cl,var_ds.ps,var_ds.ap,var_ds.b)
    if ds_ps is None: 
        ds_ps = var_ds
    varkeys = list(var_ds.variables.keys())
    if 'a' in varkeys and 'b' in varkeys:
        var_pr=geocat.comp.interpolation.interp_hybrid_to_pressure(var_ds.cl.load(),ds_ps.ps.load(),var_ds.a.load(),var_ds.b.load(),
                                    new_levels=np.array([100000., 92500., 85000., 70000., 50000., 40000., 30000., 25000., 20000., ]))
    elif 'ap' in varkeys and 'b' in varkeys:
        # ap = a * P0
        var_pr=geocat.comp.interpolation.interp_hybrid_to_pressure(var_ds.cl.load(),ds_ps.ps.load(),var_ds.ap.load()/1000.0e2,var_ds.b.load(),
                                    new_levels=np.array([100000., 92500., 85000., 70000., 50000., 40000., 30000., 25000., 20000., ]))
    elif 'ptop' in varkeys:
        a = (1-var_ds.lev.load()) * var_ds.ptop.load()/1000e2
        b = var_ds.lev.load()
        var_pr=geocat.comp.interpolation.interp_hybrid_to_pressure(var_ds.cl.load(),ds_ps.ps.load(),a,b,
                                    new_levels=np.array([100000., 92500., 85000., 70000., 50000., 40000., 30000., 25000., 20000., ]))
    elif 'b' in varkeys and 'orog' in varkeys:
        print('Implement stuff....')

    ### get weights depending on thickness of level
    var_low=var_pr.sel(plev=slice(100000.0,70000.0))
    pressure_top = 70000.0
    pressure_levels = var_low.plev
    pressure_levels=pressure_levels.assign_attrs({'units':'Pa'})
    var_weights=var_low.copy()
    var_weights.values=geocat.comp.dpres_plevel(pressure_levels,ds_ps.ps,pressure_top).fillna(0.0).values
    ### calculate weighted mean
    var_wt_avg_low=var_low.weighted(var_weights).mean(dim='plev')
    return var_wt_avg_low

# Connect to AWS S3 storage
fs = s3fs.S3FileSystem(anon=True)

## By downloading the master CSV file enumerating all available data stores, we can interact with the spreadsheet
## through a pandas DataFrame to search and explore for relevant data using the CMIP6 controlled vocabulary:
    
df = pd.read_csv("https://cmip6-pds.s3.amazonaws.com/pangeo-cmip6.csv")

def test_preprocess(activity, experiment, modelName, institute, member, testvar):
    t1 = time.time()
    print(testvar)
    inputStr  = "activity_id=='{0}' & institution_id=='{1}' & source_id=='{2}' & table_id=='Amon' & experiment_id=='{3}' &  member_id=='{4}' & variable_id==".format(activity, institute, modelName,experiment,member)

    data = getData(inputStr + "'{0}'".format(testvar))
    Out = preprocess(data[testvar])

    Pre = xr.DataArray(name=testvar+'_pre', data=Out.data, attrs={'long_name':'Processed '+data[testvar].long_name, 'units':data[testvar].units}, coords={'time': Out.time,'lat': Out.lat,'lon': Out.lon})

    ds = Pre.to_dataset(name = testvar)
    print(time.time() - t1)
    ds.to_netcdf('/home/jupyter-haruki/work/test.nc')
    print(time.time() - t1)

def preprocess_input(activity, experiment, modelName, institute, member, lf_query = None, load_ps = False):
    with dask.config.set(schedular='threads', pool=ThreadPool(30)):
        print('start')
        print(datetime.now())

        print(experiment)
        print(datetime.now())
        inputStr  = "activity_id=='{0}' & institution_id=='{1}' & source_id=='{2}' & table_id=='Amon' & experiment_id=='{3}' &  member_id=='{4}' & variable_id==".format(activity, institute, modelName,experiment,member)

        l_var = ['rsut','rsdt','rsutcs','clivi','clwvi','cl','rlut'] #input data
        if load_ps: l_var.append('ps')
        data = {var:getData(inputStr+"'{0}'".format(var)) for var in l_var}

        if lf_query is None:
            inputStr  = "activity_id=='{0}' & institution_id=='{1}' & source_id=='{2}' & table_id=='fx' & experiment_id=='{3}' &  member_id=='{4}' & variable_id==".format(activity, institute, modelName,experiment,member)
        else:
            inputStr= lf_query
        data['sftlf'] = getData(inputStr+"'sftlf'") ## land area fraction as %

        ## Calculate CRES
        cres=(data['rsdt'].rsdt - data['rsut'].rsut) - (data['rsdt'].rsdt - data['rsutcs'].rsutcs)

        ## Calculate alpha
        alpha= data['rsut'].rsut/data['rsdt'].rsdt
        #replace nan and inf values
        alpha=alpha.where(np.isfinite(alpha),0.0)

        ## get low level clouds
        if modelName == 'CESM2-WACCM':
            ## for high top model, going to remove vertical levels above 50hPa
            # Note since this is hybrid sigma-pressure, at levels above ~100hPa, the model coords are = pressure, 
            # so we can simply select below without worrying about interpolation
            if data['cl'].lev.positive == 'up':
                lcloud=getLowCloud(data['cl'].isel(lev=(data['cl'].lev < -50)))
                lcloud=lcloud.where(np.isfinite(lcloud),0.0)
            elif data['cl'].lev.positive == 'down':
                # lev is a fraction of P0 = 1e6Pa
                lcloud=getLowCloud(data['cl'].isel(lev=(data['cl'].lev > 50e-3)))
                lcloud=lcloud.where(np.isfinite(lcloud),0.0)
        elif modelName =='FGOALS-g3':
            lcloud=getLowCloud(data['cl'].isel(lev=(data['cl'].lev > 100e-3)))
            lcloud=lcloud.where(np.isfinite(lcloud),0.0)
        elif (data['cl'].lev.units == '1' or data['cl'].lev.units == '1.0') and data['cl'].lev.positive == 'down':
            if load_ps:
                lcloud=getLowCloud(data['cl'].isel(lev=(data['cl'].lev > 100e-3)),ds_ps = data['ps'])
            else:
                lcloud=getLowCloud(data['cl'].isel(lev=(data['cl'].lev > 100e-3)))
            lcloud=lcloud.where(np.isfinite(lcloud),0.0)

        elif data['cl'].lev.units == 'Pa' and data['cl'].lev.positive == 'down':
            lcloud=getLowCloud(data['cl'].isel(lev=(data['cl'].lev > 100e2)))
            lcloud=lcloud.where(np.isfinite(lcloud),0.0)
        elif data['cl'].lev.units == 'hPa' and data['cl'].lev.positive == 'down':
            lcloud=getLowCloud(data['cl'].isel(lev=(data['cl'].lev > 100)))
            lcloud=lcloud.where(np.isfinite(lcloud),0.0)

        ### preprocess
        print('ok1')
        cresOut=preprocess(cres)
        alphaOut=preprocess(alpha)
        cliviOut=preprocess(data['clivi'].clivi)
        rlutOut=preprocess(data['rlut'].rlut)
        clwviOut=preprocess(data['clwvi'].clwvi)
        ## land-sea mask - constant broadcasted
        print('ok2')
        sftlfOut, b2 = xr.broadcast(data['sftlf'].sftlf, data['rsdt'].rsdt, exclude=('lat','lon'))
        sftlfOut=sftlfOut.where(sftlfOut==0,1) ### data has values as 0 for ocean or 100 for land sa making 100 as 1
        print('ok3')
        lcloudOut=preprocess(lcloud)
        print('ok4')

        ### create dataArrays of preprocessed var and original variables
        cres_Pre = xr.DataArray(name='cres_pre', data=cresOut.data, attrs={'long_name':'Processed Cloud radiative effect in shortwave', 'units' :'W m-2', 'Description':'(rsdt - rsut) - (rsdt - rsutcs)'},
                            coords={'time': cres.time,'lat': cres.lat,'lon': cres.lon})
        cres = xr.DataArray(name='cres', data=cres.load(), attrs={'long_name':'Cloud radiative effect in shortwave', 'units' :'W m-2', 'Description':'(rsdt - rsut) - (rsdt - rsutcs)'}, 
                            coords={'time': cres.time,'lat': cres.lat,'lon': cres.lon})

        alpha_Pre = xr.DataArray(name='alpha_pre', data=alphaOut.data, attrs={'long_name':'Processed Planetary Albedo', 'Description':' (rsut/rsdt)'}, 
                            coords={'time': alpha.time,'lat': alpha.lat,'lon': alpha.lon})
        alpha = xr.DataArray(name='alpha', data=alpha.data, attrs={'long_name':'Planetary Albedo', 'Description':' (rsut/rsdt)'}, 
                            coords={'time': alpha.time,'lat': alpha.lat,'lon': alpha.lon})
        
        clivi_Pre = xr.DataArray(name='clivi_pre', data=cliviOut.data, attrs={'long_name':'Processed '+data['clivi'].clivi.long_name, 'units':data['clivi'].clivi.units}, 
                            coords={'time': cliviOut.time,'lat': cliviOut.lat,'lon': cliviOut.lon})
        clivi = xr.DataArray(name='clivi', data=data['clivi'].clivi.load(), attrs={'long_name': data['clivi'].clivi.long_name, 'units':data['clivi'].clivi.units}, 
                            coords={'time': cliviOut.time,'lat': cliviOut.lat,'lon': cliviOut.lon})
    
        clwvi_Pre = xr.DataArray(name='clwvi_pre', data=clwviOut.data, attrs={'long_name':'Processed '+data['clwvi'].clwvi.long_name, 'units':data['clwvi'].clwvi.units}, 
                            coords={'time': clwviOut.time,'lat': clwviOut.lat,'lon': clwviOut.lon})
        clwvi = xr.DataArray(name='clwvi', data=data['clwvi'].clwvi.load(), attrs={'long_name':data['clwvi'].clwvi.long_name, 'units':data['clwvi'].clwvi.units}, 
                            coords={'time': clwviOut.time,'lat': clwviOut.lat,'lon': clwviOut.lon})
    
        rlut_Pre = xr.DataArray(name='rlut_pre', data=rlutOut.data, attrs={'long_name':'Processed '+data['rlut'].rlut.long_name, 'units':data['rlut'].rlut.units}, 
                            coords={'time': rlutOut.time,'lat': rlutOut.lat,'lon': rlutOut.lon})
        rlut = xr.DataArray(name='rlut', data=data['rlut'].rlut.load(), attrs={'long_name': data['rlut'].rlut.long_name, 'units':data['rlut'].rlut.units}, 
                            coords={'time': rlutOut.time,'lat': rlutOut.lat,'lon': rlutOut.lon})
    
        lcloud_Pre = xr.DataArray(name='lcloud_pre', data=lcloudOut.data, attrs={'long_name':'Processed low_cloud_percentage', 'Description':'cloud percentage in each layer - averaged surface to 700hPa'}, 
                            coords={'time': lcloud.time,'lat': lcloud.lat,'lon': lcloud.lon})
        lcloud = xr.DataArray(name='lcloud', data=lcloud.data, attrs={'long_name':'low_cloud_percentage', 'Description':'cloud percentage in each layer - averaged surface to 700hPa'}, 
                            coords={'time': lcloud.time,'lat': lcloud.lat,'lon': lcloud.lon})

        lsMask = xr.DataArray(name='lsMask', data=sftlfOut.load(), attrs={'long_name':'land_sea_mask', 'Description':'land fraction of each grid cell 0 for ocean and 1 for land'}, 
                            coords={'time': cres_Pre.time,'lat': cres_Pre.lat,'lon': cres_Pre.lon})
        
        # Save one DataArray as dataset
        input_ds = cres_Pre.to_dataset(name = 'cres_pre')

        # Add next DataArray to existing dataset (ds)
        input_ds['cres'] = cres
        input_ds['alpha_pre'] = alpha_Pre
        input_ds['alpha'] = alpha
        input_ds['clivi_pre'] = clivi_Pre 
        input_ds['clivi'] = clivi
        input_ds['clwvi_pre'] = clwvi_Pre
        input_ds['clwvi'] = clwvi
        input_ds['rlut_pre'] = rlut_Pre
        input_ds['rlut'] = rlut
        input_ds['lcloud_pre'] = lcloud_Pre 
        input_ds['lcloud'] = lcloud
        input_ds['lsMask'] = lsMask 
                
        print('merge1')
        print(datetime.now())
        input_ds.to_netcdf(path='/home/jupyter-haruki/work/{0}_{1}_{2}_Input.nc'.format(modelName, member, experiment),mode='w',format='NETCDF4')
        #print(path)
        print('toncdf')
        print(datetime.now())

def preprocess_output(activity,experiment, modelName, institute,member):
    print(experiment)
    print(datetime.now())
    with dask.config.set(schedular='threads', pool=ThreadPool(30)):
        inputStr  = "activity_id=='{0}' & institution_id =='{1}' & source_id=='{2}' & table_id=='Amon' & experiment_id=='{3}' &  member_id=='{4}' & variable_id==".format(activity, institute, modelName,experiment,member)
        l_var = ['tas','psl','pr','ps','evspsbl'] ## output data
        data = {var:getData(inputStr+"'{0}'".format(var)) for var in l_var}

        ### preprocess
        print('ok1')
        pslOut=preprocess(data['psl'].psl.load())
        prOut=preprocess(data['pr'].pr.load())
        tasOut=preprocess(data['tas'].tas.load())
        
        ## check for slp mass balance constraint 
        # calculate annual mean
        slp_am = pslOut.groupby('time.year').mean(dim='time').mean(dim=('lat','lon'))
        cntpsl=slp_am[abs(slp_am)>0.1].count().values
            
        psl_Pre = xr.DataArray(name='psl_pre', data=pslOut.data, attrs={'long_name':'Processed '+data['psl'].psl.long_name, 'units':data['psl'].psl.units, 'count of constraint violation':str(cntpsl)}, coords={'time': pslOut.time,'lat': pslOut.lat,'lon': pslOut.lon})
        psl = xr.DataArray(name='psl', data=data['psl'].psl.load(), attrs={'long_name': data['psl'].psl.long_name, 'units':data['psl'].psl.units}, coords={'time': pslOut.time,'lat': pslOut.lat,'lon': pslOut.lon})

        tas_Pre = xr.DataArray(name='tas_pre', data=tasOut.data, attrs={'long_name':'Processed '+data['tas'].tas.long_name, 'units':data['tas'].tas.units}, coords={'time': tasOut.time,'lat': tasOut.lat,'lon': tasOut.lon})
        tas = xr.DataArray(name='tas', data=data['tas'].tas.load(), attrs={'long_name': data['tas'].tas.long_name, 'units':data['tas'].tas.units}, coords={'time': tasOut.time,'lat': tasOut.lat,'lon': tasOut.lon})

        pr_Pre = xr.DataArray(name='pr_pre', data=prOut.data, attrs={'long_name':'Processed '+data['pr'].pr.long_name, 'units':data['pr'].pr.units}, coords={'time': prOut.time,'lat': prOut.lat,'lon': prOut.lon})
        pr = xr.DataArray(name='pr', data=data['pr'].pr.load(), attrs={'long_name': data['pr'].pr.long_name, 'units':data['pr'].pr.units}, coords={'time': prOut.time,'lat': prOut.lat,'lon': prOut.lon})

        # Save one DataArray as dataset
        output_ds = psl_Pre.to_dataset(name = 'psl_pre')

        # Add next DataArray to existing dataset (ds)
        output_ds['psl'] = psl
        output_ds['tas_pre'] = tas_Pre
        output_ds['tas'] = tas
        output_ds['pr_pre'] = pr_Pre 
        output_ds['pr'] = pr

        print('merge1')
        print(datetime.now())
        
        output_ds.to_netcdf(path='/home/jupyter-haruki/work/{0}_{1}_{2}_Output.nc'.format(modelName, member, experiment),mode='w',format='NETCDF4')
        print('toncdf')
        print(datetime.now())
        print('ok')

def preprocess_constraint(activity,experiment, modelName, institute,member):
    print(experiment)
    print(datetime.now())
    with dask.config.set(schedular='threads', pool=ThreadPool(30)):
        inputStr  = "activity_id=='{0}' & institution_id =='{1}' & source_id=='{2}' & table_id=='Amon' & experiment_id=='{3}' &  member_id=='{4}' & variable_id==".format(activity, institute, modelName,experiment,member)
        l_var = ['rlutcs','hfls','hfss','rsus','rsds','rlus','rlds','ps','huss','evspsbl'] # constraint variables
        data = {var:getData(inputStr+"'{0}'".format(var)) for var in l_var}
        print('ok1')
        Pre = {}
        orig = {}
        for var in data:
            Out = preprocess(data[var][var].load())
            Pre[var] = xr.DataArray(name=var + '_pre', data=Out.data, attrs={'long_name': data[var][var].long_name, 'units':data[var][var].units}, 
                                    coords={'time': Out.time,'lat': Out.lat,'lon': Out.lon})
            orig[var] = xr.DataArray(name=var, data=data[var][var].load(), attrs={'long_name': data[var][var].long_name, 'units':data[var][var].units}, 
                                    coords={'time': Out.time,'lat': Out.lat,'lon': Out.lon})
        
        # Save one DataArray as dataset
        var = l_var[0]
        output_ds = Pre[var].to_dataset(name = var+'_pre')
        output_ds[var] = orig[var]
        for var in l_var[1:]:
            # Add next DataArray to existing dataset (ds)
            output_ds[var+'_pre'] = Pre[var]
            output_ds[var] = orig[var]

        print('merge1')
        print(datetime.now())
        
        output_ds.to_netcdf(path='/home/jupyter-haruki/work/{0}_{1}_{2}_Constraint.nc'.format(modelName, member, experiment),mode='w',format='NETCDF4')
        print('toncdf')
        print(datetime.now())
        print('ok')

    #####
    #

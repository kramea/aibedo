## A set of functions to perform the preprocessing for the AIBEDO model
## using as input xarray datasets

# Import Libraries
import xarray as xr
import pandas as pd
import numpy as np
import json
from preprocess_parallel import *

def run_preprocess(variables_in, variables_out, out_path,
                path = '/home/jupyter-haruki/work/MERRA2/',
                attribute_path = 'variable_defs_MERRA2.json'):
    ## load data
    data = {}
    for var in variables_in:
        data[var] = xr.open_dataset(path + 'MERRA2.monthly.{0}.nc'.format(var),decode_times=False).isel(time=range(492))

    d_attr = json.load(open(attribute_path))

    processed = {}
    orig = {}
    for var in variables_out:
        if var in variables_in:
            signs = [1]
            variables = [var]
            attributes = {'long_name': data[var][var].long_name, 'units':data[var][var].units}
        elif var in d_attr:
            signs = d_attr[var]['signs']
            attributes = d_attr[var]
            variables = d_attr[var]['variables']
        else:
            print('missing variable definition in {0}'.format(attribute_path))
            continue
        if 'integral' in attributes:
            processed[var], orig[var] = calc_var_integral(data,variables[0],var,attributes = attributes)
        else:
            processed[var], orig[var] = calc_var(data, variables, signs, var,attributes = attributes)
    
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

    print('Write to file')
    output_ds.to_netcdf(path=out_path,mode='w',format='NETCDF4')

## output data
variables_in = ["TLML","PRECTOT","SLP","PS","EVAP"]
variables_out = ['tas','pr','psl','ps','evspsbl']

#variables_in = ["SLP","PS"]
#variables_out = ['psl','ps']
#out_path = '/home/jupyter-haruki/work/MERRA2/MERRA2_Output.nc'
#run_preprocess(variables_in,variables_out,out_path)

## input data
variables_in = ["SWTNT","SWTNTCLR","LWTUP","LWTUPCLR","LWGNT","SWGNT","LWGNTCLR","SWGNTCLR","HFLUX","EFLUX"]
variables_out = ['cres','crel','cresSurf','crelSurf','netTOAcs','netSurfcs','hfss','netSurfRad','netTOARad']
out_path = '/home/jupyter-haruki/work/MERRA2/MERRA2_Input_Exp8.2022Aug05.nc'
run_preprocess(variables_in,variables_out,out_path)

## add land mask
data = xr.open_dataset('/home/jupyter-haruki/work/MERRA2/MERRA2_101.const_2d_asm_Nx.00000000.nc4.nc4')
indata = xr.open_dataset(out_path)
d = (data['FRLAND'][0] + data['FRLANDICE'][0])
sftlf,b2 = xr.broadcast(d , indata.isel(time=range(492))['cres'], exclude=('lat','lon'))

lsMask = xr.DataArray(name='lsMask', data=sftlf.load(), attrs={'long_name':'land_sea_mask', 'Description':'land fraction of each grid cell 0 for ocean and 1 for land'}, coords={'time': indata.isel(time=range(492))['time'],'lat': data['FRLAND'].lat,'lon': data['FRLAND'].lon})
output = lsMask.to_dataset(name = 'lsMask')
del indata,data
output.to_netcdf(path = out_path,mode='a')

## change units for variables
data = xr.open_dataset('/home/jupyter-haruki/work/MERRA2/MERRA2_Input_Exp8.2022Aug05.nc',decode_times=False)

PS = xr.open_dataset('/home/jupyter-haruki/work/MERRA2/MERRA2.monthly.PS.nc',decode_times=True)

year_month_idx = pd.MultiIndex.from_arrays([PS['time.year'].values, PS['time.month'].values])
PS = PS.assign_coords({'year_month':('time',year_month_idx)})

PS_nodiurn = PS['PS'].groupby('year_month').sum()
PS_nodiurn = PS_nodiurn.rename({'year_month':'time'})
'''
#newdata = (data['lcloud'][:492]*100).to_dataset(name='lcloud')
#newdata['clivi'] = xr.DataArray(data['clivi'].values[:492]*PS_nodiurn.values[:492]/9.81,
#                                dims = ["time","lat","lon"],
#                               attrs=data['clivi'].attrs)
#newdata['clwvi'] = xr.DataArray(data['clwvi'].values[:492]*PS_nodiurn.values[:492]/9.81,
#                                dims = ["time","lat","lon"],
#                                attrs=data['clivi'].attrs)

#newdata = -1*(data['crel'][:492]).to_dataset(name='crel')#-1*data['crel'][:492].fillna(0)

for var in ['crel']:#['lcloud','clivi','clwvi','crel']:
    print(var)
    prep_out = preprocess(newdata[var].values[:492].copy(),data.time.values[:492])
    newdata.assign({var +'_pre':xr.DataArray(prep_out,dims = ["time","lat","lon"])})

for var in data.variables:
    if not var in newdata.variables:
        newdata[var] = data[var]
del data
newdata.to_netcdf('/home/jupyter-haruki/work/MERRA2/MERRA2_Input_Exp8.2022Aug05.nc',mode='a')
'''
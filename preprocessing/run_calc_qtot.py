## Calculate the vertical integral of humidity

import preprocess_parallel
import time
import subprocess as sp
import pandas as pd
import os
import xarray as xr
import numpy as np

path = '/home/jupyter-haruki/work/processed/'
toprocess = pd.read_csv('toprocess_sfc_cs.csv')

# CMOR data variables to ingest
in_variables_in = ['hus','ps']
# output variables (must be defined in variable_defs.json)
in_variables_out = ['ps','qtot']
#['netSurfcs','lcloud','crelSurf','cresSurf','clwvi','netTOAcs','cres','crel','clivi']

## constants
g = 9.81

omit=[9,11]
for i,_ in enumerate(toprocess['model']):
    if i in omit:
        continue
    #model, member, experiment = 'CESM2','r1i1p1f1','historical'
    model, member, experiment = toprocess.iloc[i]['model'],toprocess.iloc[i]['member'],toprocess.iloc[i]['experiment']
    if experiment == 'historical':
        years = (1850,2016)
    elif experiment == 'ssp585':
        years = (2016,2100)

    # query for getting getting landfraction 
    #(landfrac is generally only available for one experiment, which is often not the historical)
    lfquery = toprocess.iloc[i]['lfquery']

    print(model,member)

    # don't want more than 5 members for a given model
    # preprocess input
    fname = '{0}.{1}.{2}.qtot.nc'.format(model, experiment, member)
    fname_dqdt = '{0}.{1}.{2}.dqdt.nc'.format(model, experiment, member)

    if not os.path.isfile(path+fname_dqdt):
        print('Process input to ',fname)
        t1 = time.time()

        # run the preprocessing
        preprocess_parallel.run_preprocess(experiment, model,member,in_variables_in,in_variables_out,
                            attribute_path = 'variable_defs.json',out_path = path + fname,
                            lf_query=lfquery,years=None)

        ## Calculate moisture content tendency
        datain = xr.open_dataset('{0}/{1}'.format(path,fname),decode_times=False)

        # Apply unit conversion
        Q = datain['qtot'][:].values*datain['ps'][:].values/g

        # calculate time values in seconds
        if 'days' in datain.time[:].units:
            t = datain.time[:].values * 24 * 60 * 60 
        elif 'hours' in datain.time[:].units:
            t = datain.time[:].values * 60 * 60 

        # central difference
        dQ = np.copy(Q)
        dQ[1:-1] = (Q[2:] - Q[:-2])/(t[2:] - t[:-2])[:,np.newaxis,np.newaxis]
        dQ[0] = (Q[1] - Q[0])/(t[1] - t[0])
        dQ[-1] = (Q[-1] - Q[-2])/(t[1] - t[0])

        ## Write to file
        outDS = xr.DataArray(name='dqdt', data=dQ, attrs={'long_name':'Atmospheric moisture tendency', 'Description':'Time derivative of column integrated specific humidity'}, 
                                    coords={'time': datain.time,'lat': datain.lat,'lon': datain.lon}).to_dataset(name = 'dqdt')
        outDS.to_netcdf(path='{0}/{1}'.format(path, fname_dqdt),format='NETCDF4')

        # run the remapping to the icosohedral sphere
        sp.run('cdo remapbil,/home/jupyter-haruki/preprocessing/isosphere_6.txt {0}/{1} {0}/isosph.{1} '.format(path,fname_dqdt),shell=True)
        sp.run('cdo remapbil,/home/jupyter-haruki/preprocessing/isosphere_5.txt {0}/{1} {0}/isosph5.{1} '.format(path,fname_dqdt),shell=True)
        print('Walltime: ', time.time()-t1) 

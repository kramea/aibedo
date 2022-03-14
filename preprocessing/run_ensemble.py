import preprocessing
import xarray as xr
import intake
import s3fs
import numpy as np
import pandas as pd

import time

d_activity = {'historical':'CMIP',
            'ssp585':'ScenarioMIP'}
col = intake.open_esm_datastore("https://cmip6-pds.s3.amazonaws.com/pangeo-cmip6.json")
failed = []
for experiment in d_activity:
    for modelName in ['MIROC-ES2L','GFDL-ESM4','CNRM-ESM2-1','BCC-CSM2-MR','CNRM-CM6-1','GFDL-CM4']:
        query = dict(experiment_id=[experiment],
                    table_id='Amon',
                    variable_id=['cl'],
                    source_id = modelName)
        # run search
        col_subset = col.search(**query);
        #print(col_subset.df.iloc[0])
        dsets = col_subset.to_dataset_dict(zarr_kwargs={'consolidated': True},
                                    storage_options={'token': 'anon'});

        code = list(dsets.keys())[0]
        activity, institute, model, experiment, __, __ = list(dsets.keys())[0].split('.')
        a = [int(str(ens.values).split('i')[0][1:]) for ens in dsets[code].member_id 
                    if int(str(ens.values).split('i')[1].split('p')[0][:]) == 1]
        s = np.argsort(a)
        l = [str(ens.values) for ens in dsets[code].member_id]
        #print(modelName, np.array(l)[s])
        i = 0
        for member in np.array(l)[s]:
            if i >= 5:
                break;
            print(activity, experiment, modelName, institute, member)
            try:
                preprocessing.preprocess_input(activity, experiment, modelName, institute, member)
                preprocessing.preprocess_output(activity, experiment, modelName, institute, member)
            except:
                failed.append([activity, experiment, modelName, institute, member])
            i+=1
print(failed)
#for member in ['r1i1p1f1', 'r2i1p1f1', 'r3i1p1f1', 'r4i1p1f1', 'r5i1p1f1']:
#    #member = 'r1i1p1f1'
#    preprocessing.preprocess_input(activity, experiment, modelName, institute, member)
#    preprocessing.preprocess_output(activity, experiment, modelName, institute, member)

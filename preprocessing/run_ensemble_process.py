import preprocess_parallel
import time
import subprocess as sp
import pandas as pd
import os

## Script to go through pre-processing and remapping of the CMIP data

# load list of models,experiments to process
toprocess = pd.read_csv('toprocess_sfc_cs.csv')

# path to output
path = '/home/jupyter-haruki/work/processed/'

# CMOR data variables to ingest
in_variables_in = ['clwvi','clivi','rlut','rsut','rsdt','rsutcs','rlutcs','hfss','hfls',
                'rlus','rsus','rsds','rlds','rsdscs','rsuscs','rldscs','cl','ps']
# output variables (must be defined in variable_defs.json)
in_variables_out = ['netSurfcs','lcloud','crelSurf','cresSurf','clwvi',
               'netTOAcs','cres','crel','clivi']

# CMOR data variables to ingest
out_variables_in = ['tas','pr','psl','evspsbl','ps']
# input variables (since these are the same as the CMOR, we don't need to define them)
out_variables_out = ['tas','pr','psl','evspsbl','ps']

# Ensemble count for each model
d_count = {'historical':{},'ssp585':{}}

# List of indices to omit (e.g. they are already completed or some error for each model)
omit = []#[0,1,2,3,4,5,6,7,8,13,14,15,16,17,18,19,20,25,26,27]
for i,_ in enumerate(toprocess['model']):
    if i in omit:
        continue
    # get information from the toprocess list
    model, member, experiment = toprocess.iloc[i]['model'],toprocess.iloc[i]['member'],toprocess.iloc[i]['experiment']
    # query for getting landfraction 
    #(landfrac is generally only available for one experiment, which is often not the historical)
    lfquery = toprocess.iloc[i]['lfquery']

    print(model,member)
    # don't want more than 5 members for a given model
    if not model in d_count[experiment]:
        d_count[experiment][model] = 1
    if d_count[experiment][model] < 5:
        # preprocess input
        fname = '{0}.{1}.{2}.Input.netSurfcs.nc'.format(model, experiment, member)
        if not os.path.isfile(path+fname):
            print('Process input to ',fname)
            t1 = time.time()
            # run the preprocessing
            preprocess_parallel.run_preprocess(experiment, model,member,in_variables_in,in_variables_out,
                                attribute_path = 'variable_defs.json',out_path = path + fname,
                                lf_query=lfquery)

            # run the remapping to the icosohedral sphere
            sp.run('cdo remapbil,/home/jupyter-haruki/preprocessing/isosphere_6.txt {0}/{1} {0}/isosph.{1} '.format(path,fname),shell=True)
            sp.run('cdo remapbil,/home/jupyter-haruki/preprocessing/isosphere_5.txt {0}/{1} {0}/isosph5.{1} '.format(path,fname),shell=True)
            print('Walltime: ', time.time()-t1)

        # preprocess output
        fname = '{0}.{1}.{2}.Output.nc'.format(model, experiment, member)
        if not os.path.isfile(path+fname):
            print('Process output to ',fname)
            t1 = time.time()
            # run the preprocessing
            preprocess_parallel.run_preprocess(experiment, model,member,out_variables_in,out_variables_out,
                                    attribute_path = 'variable_defs.json',out_path = path + fname,lf_query=lfquery,
                                    append=True)
            # run the remapping to the icosohedral sphere
            sp.run('cdo remapbil,/home/jupyter-haruki/preprocessing/isosphere_6.txt {0}/{1} {0}/isosph.{1} '.format(path,fname),shell=True)
            sp.run('cdo remapbil,/home/jupyter-haruki/preprocessing/isosphere_5.txt {0}/{1} {0}/isosph5.{1} '.format(path,fname),shell=True)
           print('Walltime: ', time.time()-t1)
        d_count[experiment][model] += 1

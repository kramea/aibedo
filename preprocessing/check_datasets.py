import xarray as xr
import json
import pandas as pd
import numpy as np
import os
## quick script to run through data and check if the global mean values look okay (e.g. no NaN)
toprocess = pd.read_csv('toprocess_sfc_cs.csv')
path = '/home/jupyter-haruki/work/processed/'
in_variables_out = ['lcloud','crelSurf','cresSurf','clwvi','netSurf','netTOA',
                'netTOAcs','cres','crel','clivi']
out_variables_out = ['tas','pr','psl','evspsbl']

indata = {"model":[],"experiment":[],"ensmem":[],"grid":[],"nyears":[]}
for var in in_variables_out:
    indata[var] = []
    indata[var+'_pre'] = []
indata['lsMask'] = []
outdata = {"model":[],"experiment":[],"ensmem":[],"grid":[],"nyears":[]}
for var in out_variables_out:
    outdata[var] = []
    outdata[var+'_pre'] = []
l_complete = []
l_fixland = []
for i in toprocess.index:
    model, member, experiment = toprocess.iloc[i]['model'],toprocess.iloc[i]['member'],toprocess.iloc[i]['experiment']
    print(model, member,experiment)
    fnamein = '{0}.{1}.{2}.Input.Exp8.nc'.format(model, experiment, member)
    sph_fnamein = 'isosph.{0}.{1}.{2}.Input.Exp8.nc'.format(model, experiment, member)
    fnameout = '{0}.{1}.{2}.Output.nc'.format(model, experiment, member)
    sph_fnameout = 'isosph.{0}.{1}.{2}.Output.nc'.format(model, experiment, member)
    
    print(fnamein)
    if os.path.isfile(path+fnamein):
        try:
            datain = xr.open_dataset(path + fnamein)
        except:
            continue
        indata["model"].append(model)
        indata["experiment"].append(experiment)
        indata["ensmem"].append(member)
        complete = True
        indata["grid"].append('latlon')
        for var in in_variables_out:
            print(var)
            if not var in datain.variables:
                complete=False
                indata[var].append('missing')
                indata[var+'_pre'].append('missing')
            else:
                if np.mean(datain[var][0].values) == 0 or np.isnan(np.mean(datain[var][0].values)):
                    print("Incomplete", model, experiment,member, var)
                    complete = False
                indata[var].append(np.mean(datain[var][0].values))
                indata[var+'_pre'].append(np.mean(datain[var+'_pre'][0].values))
        if np.mean(datain['lsMask'][0].values) == 0 or np.isnan(np.mean(datain['lsMask'][0].values)):
            print("Incomplete", model, experiment,member, var)
            complete = False
            l_fixland.append([i,model,experiment,member])
        indata['lsMask'].append(np.mean(datain['lsMask'][0].values))

        indata["nyears"].append(datain.time.shape[0])

        datain = xr.open_dataset(path + sph_fnamein)
        indata["model"].append(model)
        indata["experiment"].append(experiment)
        indata["ensmem"].append(member)
        indata["grid"].append('isosphere')
        for var in in_variables_out:
            if not var in datain.variables:
                indata[var].append('missing')
                indata[var+'_pre'].append('missing')
            else:
                indata[var].append(np.mean(datain[var][0].values))
                indata[var+'_pre'].append(np.mean(datain[var+'_pre'][0].values))
        indata["nyears"].append(datain.time.shape[0])
        indata['lsMask'].append(np.mean(datain['lsMask'][0].values))

        try:
            datain = xr.open_dataset(path + fnameout)
        except:
            continue;
        outdata["model"].append(model)
        outdata["experiment"].append(experiment)
        outdata["ensmem"].append(member)
        outdata["grid"].append('latlon')
        for var in out_variables_out:
            if not var in datain.variables:
                outdata[var].append('missing')
                outdata[var+'_pre'].append('missing')
            else:
                outdata[var].append(np.mean(datain[var][0].values))
                outdata[var+'_pre'].append(np.mean(datain[var+'_pre'][0].values))
        outdata["nyears"].append(datain.time.shape[0])

        datain = xr.open_dataset(path + sph_fnameout)
        outdata["model"].append(model)
        outdata["experiment"].append(experiment)
        outdata["ensmem"].append(member)
        outdata["grid"].append('isosphere')
        for var in out_variables_out:
            if not var in datain.variables:
                outdata[var].append('missing')
                outdata[var+'_pre'].append('missing')
            else:
                outdata[var].append(np.mean(datain[var][0].values))
                outdata[var+'_pre'].append(np.mean(datain[var+'_pre'][0].values))
        outdata["nyears"].append(datain.time.shape[0])
    else:
        complete=False
    if complete:
        l_complete.append([i,model,experiment,member])
print(l_complete)
print(l_fixland)
outdf = pd.DataFrame(outdata)
indf = pd.DataFrame(indata)
outdf.to_csv('checklist_outdata.csv')
indf.to_csv('checklist_indata.csv')
import numpy as np
import xarray as xr
import torch
from pathlib import Path
import re
import time


input_file = '/Users/kramea/Documents/AIBEDO_dir/data_aibedo/compress.isosph.CESM2.historical.r1i1p1f1.Input.Exp8_fixed.nc'
output_file = '/Users/kramea/Documents/AIBEDO_dir/data_aibedo/compress.isosph.CESM2.historical.r1i1p1f1.Output.nc'

inDS = xr.open_dataset(input_file)
outDS = xr.open_dataset(output_file)
n_pixels = len(inDS.ncells)
time_length = 4

in_vars = [ 'crelSurf_pre', 'crel_pre', 'cresSurf_pre', 'cres_pre', 'netTOAcs_pre', 'lsMask', 'netSurfcs_pre']
out_vars = ['tas_pre', 'psl_pre', 'pr_pre']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet = torch.load('/Users/kramea/Documents/AIBEDO_dir/data_aibedo/sunet_timelag_model.pt',
                        map_location=torch.device('cpu'))
unet = unet.module.to(device)
unet.eval()

modelfilename = Path(output_file).stem
p = re.compile('compress.isosph.(.*).historical.r1i1p1f1.Output')
modelname = p.findall(modelfilename)[0]

data_all = []
for var in in_vars:
    temp_data = np.reshape(np.concatenate(inDS[var].data, axis=0), [-1, n_pixels, 1])
    data_all.append(temp_data)
dataset_in = np.concatenate(data_all, axis=2)

data_all = []
for var in out_vars:
    temp_data = np.reshape(np.concatenate(outDS[var].data, axis=0), [-1, n_pixels, 1])
    data_all.append(temp_data)
dataset_out = np.concatenate(data_all, axis=2)

new_in_data = []
new_out_data = []
for i in range(0, len(dataset_in)-time_length):
    intemp = np.concatenate(dataset_in[i:i + time_length, :, :], axis=1)
    new_in_data.append(intemp)
    new_out_data.append(dataset_out[i+time_length-1, :, :])


dataset_in_lstm = np.asarray(new_in_data)
dataset_out_lstm = np.asarray(new_out_data)

inPredict = dataset_in_lstm[0:1]
outPredict = dataset_out_lstm[0:1]

preds = unet(torch.Tensor(inPredict))

pred_numpy = preds.detach().cpu().numpy()

np.save(("/Users/kramea/Documents/AIBEDO_dir/data_aibedo/"+ modelname + "_predictions.npy"), pred_numpy)
np.save(("/Users/kramea/Documents/AIBEDO_dir/data_aibedo/" + modelname + "_groundtruth.npy"), outPredict)




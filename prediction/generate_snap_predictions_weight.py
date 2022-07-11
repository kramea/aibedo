import numpy as np
import xarray as xr
import torch
from spherical_unet.models.spherical_unet.unet_model import SphericalUNet
from pathlib import Path
import re
import time


input_file = 'compress.isosph.CESM2.historical.r1i1p1f1.Input.Exp8_fixed.nc'
output_file = 'compress.isosph.CESM2.historical.r1i1p1f1.Output.nc'

inDS = xr.open_dataset(input_file)
outDS = xr.open_dataset(output_file)
n_pixels = len(inDS.ncells)

in_vars = [ 'crelSurf_pre', 'crel_pre', 'cresSurf_pre', 'cres_pre', 'netTOAcs_pre', 'lsMask', 'netSurfcs_pre']
out_vars = ['tas_pre', 'psl_pre', 'pr_pre']

weights_file = torch.load('sunet_state_6.pt', 
                          map_location=torch.device('cpu'))
weights_file = {key.replace("module.", ""): value for key, value in weights_file.items()}

unet = SphericalUNet('icosahedron', n_pixels, 6, 'combinatorial', 3, 7, 3)

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

unet.load_state_dict(weights_file, strict=False)
unet.eval()

inPredict = dataset_in[0:1]
outPredict = dataset_out[0:1]

preds = unet(torch.Tensor(inPredict))

pred_numpy = preds.detach().cpu().numpy()

np.save((modelname + "_predictions.npy"), pred_numpy)
np.save((modelname + "_groundtruth.npy"), outPredict)



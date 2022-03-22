import numpy as np
import os
from data_loader import load_ncdf, normalize, load_ncdf_to_SphereIcosahedral
import glob
from pathlib import Path
from tqdm import tqdm

data_folder = '/home/ubuntu/'
npy_folder = '/home/ubuntu/npy_files/'
# Provide input variables after finalizing
input_vars = ['cres_pre', 'crel_Pre', 'netTOAcs_Pre', 'clivi_pre', 'clwvi_pre', 'cresSurf_Pre', 'crelSurf_Pre']
output_vars = ['tas_pre', 'pr_pre', 'psl_pre']
depth = 6 # You can try this with 1 to test

for name in tqdm(glob.glob(data_folder+'*.nc')):
    print("Processing .nc file: ", name)
    fname = npy_folder + Path(name).stem  + "_" + str(depth) + ".npy"
    print("Checking if npy file exists..")
    if os.path.exists(fname):
        print("Gridded .npy file exists")
    else:
        print("Generating .npy file ", fname, "at grid level", depth, "...")
        if "Input" in name:
            lon_list, lat_list, dset = load_ncdf_to_SphereIcosahedral(name, depth, input_vars)
        else:
            lon_list, lat_list, dset = load_ncdf_to_SphereIcosahedral(name, depth, output_vars)
        np.save(fname, dset)







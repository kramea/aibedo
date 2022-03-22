import numpy as np
import os
from data_loader import load_ncdf, normalize, load_ncdf_to_SphereIcosahedral


in_vars = ['cres_pre', 'crel_Pre', 'netTOAcs_Pre', 'clivi_pre', 'clwvi_pre','lcloud_pre', 'cresSurf_Pre', 'crelSurf_Pre', 'netSurfcs_Pre']
lon_list, lat_list, dataset = load_ncdf_to_SphereIcosahedral('/home/ubuntu/Exp7_CESM2_r1i1p1f1_historical_Input.nc', glevel=3, dvar=in_vars)

np.save("Exp7_CESM2_r1i1p1f1_historical_Input_3.npy", dataset)



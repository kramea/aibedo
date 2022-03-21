import numpy as np
import os
from data_loader import load_ncdf, normalize, load_ncdf_to_SphereIcosahedral

lon_list, lat_list, dataset = load_ncdf_to_SphereIcosahedral('/home/ubuntu/Exp3_CESM2_r1i1p1f1_historical_Input.nc', glevel=1)

np.save("test.npy", dataset)



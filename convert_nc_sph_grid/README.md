## Install this one by one

```
conda create --name spherical python=3.7
source activate spherical
pip install git+https://github.com/epfl-lts2/pygsp.git@39a0665f637191152605911cf209fc16a36e5ae9#egg=PyGSP
pip install git+https://github.com/deepsphere/deepsphere-pytorch
conda install  pandas h5py xarray dask netCDF4 bottleneck
pip install torchsummary
pip install tqdm
conda install -c conda-forge trimesh
```
## Then run this script

```
python input_generation.py
```


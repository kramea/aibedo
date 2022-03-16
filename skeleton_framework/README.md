#install
```
conda create --name deepsphere python=3.7
source activate deepsphere
pip install git+https://github.com/epfl-lts2/pygsp.git@39a0665f637191152605911cf209fc16a36e5ae9#egg=PyGSP
conda install pytorch=1.3.1 torchvision=0.4.2 cudatoolkit=10.0 -c pytorch
pip install git+https://github.com/deepsphere/deepsphere-pytorch
conda install -c conda-forge cartopy
conda install trimesh cpandas torchsummary h5py xarray dask netCDF4 bottleneck
```
#Execute code
```
python main_sphericalunet.py --config-file config.example.yml
```
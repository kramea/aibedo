#install
```
conda create --name deepsphere python=3.7
source activate deepsphere
pip install git+https://github.com/epfl-lts2/pygsp.git@39a0665f637191152605911cf209fc16a36e5ae9#egg=PyGSP
conda install pytorch=1.3.1 torchvision=0.4.2 cudatoolkit -c pytorch
pip install git+https://github.com/deepsphere/deepsphere-pytorch
conda install -c conda-forge cartopy
conda install  pandas h5py xarray dask netCDF4 bottleneck
pip install torchsummary
conda install -c conda-forge trimesh

```
#Execute code
```
python main_sphericalunet.py --config-file config.example.yml
```

##Data download
download data at
```
https://parc.sharepoint.com/sites/AIBEDO/Shared%20Documents/Forms/AllItems.aspx?isAscending=false&id=%2Fsites%2FAIBEDO%2FShared%20Documents%2FGeneral%2FPreprocessed%20Data%2FExp%20Runs&sortField=Modified&viewid=91413d38%2D1020%2D42de%2Db861%2Dde367c3c550c
```
and save in data folder
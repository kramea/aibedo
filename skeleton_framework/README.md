## Installation (for CPU)

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
conda install ignite -c pytorch-nightly
```

## Installation (for GPU)

```
conda create --name deepsphere python=3.7
source activate deepsphere
pip install git+https://github.com/epfl-lts2/pygsp.git@39a0665f637191152605911cf209fc16a36e5ae9#egg=PyGSP
pip install torch==1.5.1+cu101 -f https://download.pytorch.org/whl/cu101/torch-1.5.1%2Bcu101-cp37-cp37m-linux_x86_64.whl
pip install git+https://github.com/deepsphere/deepsphere-pytorch
conda install -c conda-forge cartopy
conda install  pandas h5py xarray dask netCDF4 bottleneck
pip install torchsummary
conda install -c conda-forge trimesh
pip install pillow==6.1
conda install ignite -c pytorch-nightly
```

## Execute Spheircal-Unet

```
python sunet_compress_gpu.py --config-file config_yml/newconfig.compress.yml
```
## Execute Spherical-Conv-LSTM

```
python sunetlstm_compress_gpu.py  --config-file config_yml/newconfig.compress.yml
```
## Data

Sample data is already available in the AWS instance. Other datasets are located in S3 Bucket.

## Model file

Temporal spherical U-net model weights file trained with 3-months input (t0, t1, t2) and output (t3) at level-6 spherical grid can be found in S3 Bucket named as unet_state_6.pt. 

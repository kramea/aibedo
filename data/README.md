# Data

Please put all required files into a single data folder/directory, ``DATA_DIR``, and read through the following instructions carefully.

***Please override the data directory with either of the options below:***
   1) From the command line, use the flag: ``datamodule.data_dir=DATA_DIR``
   2) Create a config file called [configs/local/](configs/local)default.yaml adapted 
  from e.g. this example local config [configs/local/example_local_config.yaml](configs/local/example_local_config.yaml) and edit the ``datamodule.data_dir`` entry.
  Its values will be automatically used whenever you run any script.
   3) Directly edit the ``datamodule.data_dir`` entry in the [base_data_config.yaml config file](configs/datamodule/base_data_config.yaml). 

## Minimal data requirements
The data directory should contain, at minimum, the following files:

- ``ymonmean.1980_2010.compress.isosph.CMIP6.historical.ensmean.Output.PrecipCon.nc``
- ``ymonstd.1980_2010.compress.isosph.CMIP6.historical.ensmean.Output.PrecipCon.nc``
- ``CMIP6_Precip_clim_err.isoph6.npy``
- ``CMIP6_PS_clim_err.isoph6.npy``
- ``CMIP6_PE_clim_err.isoph6.npy``
- ``compress.isosph.CESM2.historical.r1i1p1f1.Input.Exp8_fixed.nc``
- ``compress.isosph.CESM2.historical.r1i1p1f1.Output.PrecipCon.nc``

You can use the provided script [download_data_minimal.sh](download_data_minimal.sh) 
to download the data from AWS (requires credentials) with the following command:
<br>**Note:** Edit the ``DATA_DIR`` flag in the script to download to a different folder.

    bash download_data_minimal.sh

**Note:** The above assumes that you want to train or evaluate on CESM2.
If you want to train or evaluate on another ESM, please also download the corresponding input and output files 
(simply replace CESM2 with any other ESM name in the two last filenames above).
You can do so programmatically by editing the ``ESMs`` list in the [download_data_minimal.sh](download_data_minimal.sh) script.

### Even more minimal data requirements

You can retain most functionality of the codebase without the ``CMIP6_<ERR>_clim_err.isoph6.npy`` files.
If you don't have these files available, you can simply set the ``model.use_auxiliary_vars=False`` flag.
Note that you won't be able to train the model with physics constraints in this case.


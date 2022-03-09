.. _aibedo_datasets:


Datasets
========

We will be training on selected Global Circulation Model (GCM) data from the Coupled Model Intercomparison Project Phase 6 (CMIP6). The training is performed on normalized anomalies for historical (1850 to 2015) and SSP5-8.5 (2016 to 2100) simulations from the following GCMs:

- NCC NorESM1-MM
- IPSL IPSL-CM6A-LR
- NCAR CESM2
- NCAR CESM2-WACCM
- MOHC UKESM1-0-LL
- E3SM-Project E3SM-1-0
- MPI MPI-ESM1-2-LR
- MIROC MIROC6

Validation and testing is performed using reanalysis data such as the [ERA5](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5) and [MERRA](https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/).

We utilize monthly averaged data and preprocess it in three steps:
1. Remove seasonal cycle
2. Detrend data using a order-3 polynomial fit separately for each month
3. Calculate the anomaly and normalize (divide by standard deviation) in a 31-year rolling window for each month separately

The model/reanalysis variables used are divided into three categories: Input, Output, and Constraints (see [CMIP6 variables codes](http://clipc-services.ceda.ac.uk/dreq/mipVars.html)).

| Category   | Variable | Description                                                        |
| ---------- | -------- | ------------------------------------------------------------------ |
| Input      | alpha    | Atmospheric Albedo (rsut/rsdt)                                     |
| Input      | cres     | Shortwave cloud Radiative effect ((rsdt - rsut) - (rsdt - rsutcs)) |
| Input      | clivi    | Mass of cloud ice water in a column                                |
| Input      | clwvi    | Mass of cloud condensed water in a column                          |
| Input      | lowcloud | Cloud concentrations integrated between 1000hPa and 700hPa         |
| Output     | tas      | 2-meter air temperature                                            |
| Output     | pr       | Precipitation                                                      |
| Output     | psl      | Sea level pressure                                                 |
| Constraint | evspsbl  | Evaporation                                                        |
| Constraint | huss     | Surface specific humidity                                          |
| Constraint | hfls     | Surface Latent heat flux                                           |
| Constraint | hfss     | Surface Sensible heat flux                                         |
| Constraint | rsus     | Surface upward shortwave flux                                      |
| Constraint | rsds     | Surface downward shortwave flux                                    |
| Constraint | rlus     | Surface upward longwave flux                                       |
| Constraint | rlds     | Surface downward longwave flux                                     |

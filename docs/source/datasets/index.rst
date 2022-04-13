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

The model/reanalysis variables used are divided into three categories: Input, Output, and Constraints (see `CMIP6 variables codes <http://clipc-services.ceda.ac.uk/dreq/mipVars.html>`

.. list-table:: Variable list and descriptions
   :widths: 20 20 60
   :header-rows: 1

   * - Category
     - Variable
     - Description
   * - Input
     - clwvi
     - Mass of cloud liquid water in a column
   * - Input
     - clivi
     - Mass of cloud ice water in a column 
   * - Input
     - cres
     - TOA Cloud radiative effect in shortwave
   * - Input
     - cresSurf
     - Surface Cloud radiative effect in shortwave
   * - Input
     - crel
     - TOA Cloud radiative effect in longwave
   * - Input
     - crelSurf
     - Surface Cloud radiative effect in longwave
   * - Input
     - netTOA
     - Net TOA radiation (all-sky) 
   * - Input
     - netTOAcs 
     - TOA radiation without clouds (clear-sky)
   * - Input
     - netSurf
     - Net Surface radiation
   * - Input
     - netSurfcs
     - Net Clearsky Surface radiation
   * - Input
     - lcloud
     - Cloud fraction averaged between 1000 hPa and 700 hPa
   * - Output
     - tas
     - 2-metre air temperature
   * - Output
     - psl
     - Sea level pressure
   * - Output
     - pr
     - Precipitation
   * - Constraint
     - ps 
     - Surface pressure
   * - Constraint
     - evspsbl
     - Evaporation
   * - Constraint
     - heatconv
     - Convergence of vertically integrated heat flux

Data required for Physics Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
To strengthen the generalisability of the AiBEDO model, we include "weak" physics-based constraints on the output to penalize unphysical results from the model.

1. Atmospheric mass balance : global, annual mean sea level pressure (psl) is equal to zero
2. Atmospheric moisture balance : global, annual mean precipitation minus evaporation (pr - evspsbl) is equal to zero
3. Precipitation low bound : Precipitation cannot be less zero
4. Local Tropical Atmosphere moisture budget : Precipitation + sensible heat flux = atmospheric cooling (pr + hfss - (rsus + rlus - rsds - rlds + rsdt - rlut) = 0)
5. Energy Balance : (forcing = feedbacks + ET convergence + storage)  
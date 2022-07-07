.. _aibedo_datasets:

Datasets
========

Training data
--------------

Our training data for Phase 1 consists of a subset of CMIP6 Earth System Model (ESM) outputs which had sufficient data availability on AWS to calculate the requisite input variables for our analysis (shown in Table 1. For each ESM, there are three sets of data hyper-cubes: (a) input, (b) output, and (c) data for enforcing physics constraints. Based on the initial results from our alpha hybrid model, we revised and increased the list of input variables to achieve better hybrid model performance. The updated list of input, output, and constraint variables is shown in Table 2.

.. list-table:: Table 1. Earth System Model datasets for Phase 1 training
   :widths: 20 20 20 20 20
   :header-rows: 1


   * - Model
     - N Historical
     - N ssp585
     - Lat spacing (deg)
     - Lon spacing (deg)
   * - CESM2
     - 1 
     - 0 
     - 0.942 
     - 1.2500
   * - CESM2-FV2 
     - 1 
     - 0 
     - 1.895
     - 2.5000
   * - CESM2-WACCM
     - 1 
     - 0 
     - 0.942 
     - 1.2500
   * - CESM2-WACCM-FV2
     - 1
     - 0
     - 1.895
     - 2.5000
   * - CMCC-CM2-SR5
     - 1
     - 0
     - 0.942
     - 1.2500
   * - CanESM5
     - 5
     - 0
     - 2.789
     - 2.8125
   * - E3SM-1-1
     - 1
     - 0
     - 1.000
     - 1.0000
   * - E3SM-1-1-ECA
     - 1
     - 0
     - 1.000
     - 1.0000
   * - FGOALS-g3
     - 2
     - 1
     - 2.278
     - 2.0000
   * - GFDL-CM4
     - 1 
     - 1
     - 1.000
     - 1.2500
   * - GFDL-ESM4
     - 1
     - 1
     - 1.000
     - 1.2500
   * - GISS-E2-1-H
     - 1
     - 0
     - 2.000
     - 2.5000
   * - MIROC-ES2L
     - 3
     - 0
     - 2.789
     - 2.8125
   * - MIROC6
     - 1
     - 0
     - 1.400
     - 1.4062
   * - MPI-ESM-1-2-HAM
     - 1
     - 0
     - 1.865
     - 1.8750
   * - MPI-ESM1-2-HR
     - 1
     - 0
     - 0.935
     - 0.9375
   * - MPI-ESM1-2-LR
     - 1
     - 0
     - 1.865
     - 1.8750
   * - MRI-ESM2-0
     - 1
     - 0
     - 1.121
     - 1.1250
   * - SAM0-UNICON
     - 1
     - 0
     - 0.942
     - 1.2500


.. list-table:: Table 2. Variable list and descriptions
   :widths: 20 20 60
   :header-rows: 1

   * - Category
     - Variable
     - Description
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
     - netTOAcs 
     - TOA radiation without clouds (clear-sky)
   * - Input
     - netSurfcs
     - Net Clearsky Surface radiation
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

The ESM data are pooled together to form the training and testing datasets for our hybrid model. However, it is important to note there are substantial differences in the climatologies and variability of some of the chosen input variables across models (Figure 1). In particular, global average cloud liquid water content, cloud ice water content, and net top of atmosphere radiation vary more across ESMs than other variables. The former two are the result of differences in cloud parameterizations between ESMs, while the latter is likely due to uncertainties in the overall magnitude of anthropogenic forcing over the historical period. Comparing spatial correlation scores (Figure 2), shows net TOA radiation fields are very similar across models while the spatial pattern of cloud ice and water content varies substantially. Such variations represent the inter-ESM uncertainty in the representation of the climate. However, many of these ESM differences are largely removed during preprocessing described below.

.. figure::
	images/box_mm_spread_1.png

	Figure 1. Box plot of spread of ESM global and time means of selected input, output, and constraint variables divided by their respective multi-model ensemble mean. Ensemble mean values shown in parentheses in the x-labels

.. figure::
	images/model_var_spatcorr_2.png

	Figure 2. Pearson-R spatial correlations between ESM time average and ESM ensemble mean fields (for data remapped to level 5 Sphere-Icosahedral grid) across the models (rows) and variables (columns), showing the inter-ESM uncertainty in the climatologies of the selected input, output, and constraint variables

Reanalysis
-----------
In addition to ESM data, we also employ "reanalysis" datasets as validation datasets. Namely the `Modern-Era Retrospective analysis for Research and Applications, Version 2 (MERRA2) <https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/>`_ and the `ECMWF Reanalysis version 5 (ERA5) <https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5>`_.
Reanalyses are models which ingest large quantities observational data to estimate the historical evolution of the atmosphere, thus providing an estimate of a wide range of atmospheric variables over the entire globe.
While these data are exactly the same as observational data, they are the best method of obtaining physically consistent and complete climate data representing the recent past of the Earth's atmosphere.
MERRA2 includes data from 1980 to 2020 at 0.5 degree resolution and ERA5 includes data from 1979 to 2021 at 0.25 degree resolution.

Preprocessing
--------------

Each of the above data hyper-cubes are preprocessed before ingestion into the hybrid model as follows:

#. **Remove seasonal cycle or "Deseasonalizing"**: We perform this process to remove any trends in the season to prepare a seasonal stationary time series data. 
#. **Remove trend or Detrend**: We fit a third degree polynomial to remove any trend in data over time. This removes secular trends (for example, rising temperatures as atmospheric CO$_2$ increases) and allows the model to be trained on fluctuations due to internal variability, rather than the forced response. 
#. **Normalized anomalies**: The anomaly at each grid point is calculated relative to a running mean that is computed over a centered 30-year window for that grid point and month. Anomalies are normalized by dividing by the standard deviation of the anomaly over the same 30-year window for that grid point and month.
#. **Remap data to Sphere-Icosahedral**: Use `Climate Data Operators <https://code.mpimet.mpg.de/projects/cdo/embedded/index.html#x1-6460002.12.1>`_ to bilinearly remap disparate ESM grids to uniform level-6 Sphere-Icosahedral grid.

.. figure::
	images/preprocessing_example.png

  Figure 3. Example preprocessing for a surface air temperature data point.



References
-----------
Allen, M. R., & Ingram, W. J. (2002). Constraints on future changes in climate and the hydrologic cycle. Nature, 419(6903), 228-232.


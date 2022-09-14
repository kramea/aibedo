#!/bin/bash
############ PLEASE ADJUST BELOW WHERE TO DOWNLOAD THE DATA ############
DATA_DIR="."  # /work/data"
######################################################################

# Download the data
ESMs=("CESM2")
######## IF YOU WANT TO DOWNLOAD data for all ESMS UNCOMMENT THE FOLLOWING LINES
# ESMs=("CESM2" "CESM2-FV2" "CESM2-WACCM" "CESM2-WACCM-FV2" "CMCC-CM2-SR5" "CanESM5" "E3SM-1-1" "E3SM-1-1-ECA" "FGOALS-g3" "GFDL-CM4" "GFDL-ESM4" "GISS-E2-1-H" "MIROC-ES2L" "MIROC6" "MPI-ESM-1-2-HAM" "MPI-ESM-1-2-HR" "MPI-ESM-1-2-LR" "MRI-ESM2-0" "SAM0-UNICON")

for f in  ${ESMs[*]};
do
  FILE_IN=compress.isosph.${f}.historical.r1i1p1f1.Input.Exp8_fixed.nc
  FILE_OUT=compress.isosph.${f}.historical.r1i1p1f1.Output.nc

  # In files
  if ! test -f "${DATA_DIR}/${FILE_IN}"; then
    aws s3 cp s3://darpa-aibedo/${FILE_IN} $DATA_DIR
  fi

  # Out files
  if ! test -f "${DATA_DIR}/${FILE_OUT}"; then
    aws s3 cp s3://darpa-aibedo/${FILE_OUT} $DATA_DIR
  fi

done

# Download statistics
MEAN_STATS="ymonmean.1980_2010.compress.isosph.CMIP6.historical.ensmean.Output.PrecipCon.nc"
STD_STATS="ymonstd.1980_2010.compress.isosph.CMIP6.historical.ensmean.Output.PrecipCon.nc"

if ! test -f "${DATA_DIR}/${MEAN_STATS}"; then
  aws s3 cp s3://darpa-aibedo/${MEAN_STATS} $DATA_DIR
fi

if ! test -f "${DATA_DIR}/${STD_STATS}"; then
  aws s3 cp s3://darpa-aibedo/${STD_STATS} $DATA_DIR
fi

# Download physics constraints auxiliary error files
for ERR in "PE_clim_err" "PS_clim_err" "Precip_clim_err";
do
  if ! test -f "${DATA_DIR}/CMIP6_${ERR}.isoph.npy"; then
    aws s3 cp s3://darpa-aibedo/CMIP6_${ERR}.isoph.npy $DATA_DIR
  fi
done


#aws s3 cp s3://darpa-aibedo/compress.isosph.MERRA2_Input_Exp8_fixed.nc data/
#aws s3 cp s3://darpa-aibedo/compress.isosph5.MERRA2_Output.nc data/
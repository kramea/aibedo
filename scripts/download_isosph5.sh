#!/bin/bash

data_dir="/work/data"
for esm in "MRI-ESM2-0" "CESM2-WACCM" "MPI-ESM-1-2-HAM" "CESM2" "GFDL-ESM4";
do
  echo $esm
  aws s3 cp s3://darpa-aibedo/compress.isosph5.${esm}.historical.r1i1p1f1.Input.Exp8_fixed.nc $data_dir
  aws s3 cp s3://darpa-aibedo/compress.isosph5.${esm}.historical.r1i1p1f1.Output.PrecipCon.nc $data_dir
  aws s3 cp s3://darpa-aibedo/isosph5.denorm_nonorm.${esm}.historical.r1i1p1f1.Output.nc $data_dir
  aws s3 cp s3://darpa-aibedo/isosph5.denorm_nonorm.${esm}.historical.r1i1p1f1.Output.nc $data_dir
done

for err in "PS" "PE" "Precip";
do
  aws s3 cp s3://darpa-aibedo/CMIP6_${err}_clim_err.isoph5.npy $data_dir
done
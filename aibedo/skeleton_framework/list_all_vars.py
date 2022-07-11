import argparse
import xarray as xr

parser = argparse.ArgumentParser(description="Provide nc file")

parser.add_argument('-f', type=str, required=True)

args = parser.parse_args()

ds = xr.open_dataset(args.f)

print(list(ds.keys()))

import xarray as xr
import matplotlib.pyplot as plt
from aibedo.utilities.plotting import animate_snapshots

# ---------------> Edit this directory pointer!
PDIR = "C:/Users/salva/PycharmProjects/Data/aibedo/xarrays"

ds_mlp = xr.load_dataset(f"{PDIR}/MLP-no-constraints-37z9auhs.nc")
ds_sunet = xr.load_dataset(f"{PDIR}/SUNet-no-constraints-2frxrgnq.nc")
ds_all_constraints_mlp = xr.load_dataset(f"{PDIR}/MLP-with-constraints-2ymsmghz.nc")

# Select which of the above datasets to plot:
ds = ds_mlp

# Select which variable to plot:
vars_to_plot = 'all'  # alternative: ['tas']

vars_to_plot = ds_mlp.variable_names.split(";") if vars_to_plot == "all" else vars_to_plot
# Show GIFs by target variable, 1by1
for var in vars_to_plot:
    var_id = var.replace('_targets', '').replace('_preds', '')
    print(f"Plotting {var_id}", var)
    ds[var_id + '_targets'] = ds_mlp[var_id + '_targets']
    ani1 = animate_snapshots(ds, var_to_plot=var_id + '_targets')
    ani2 = animate_snapshots(ds, var_to_plot=var_id + '_preds')
    plt.show()

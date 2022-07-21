import xarray as xr
import matplotlib.pyplot as plt
from aibedo.utilities.plotting import animate_snapshots

# ---------------> Edit this directory pointer!
PDIR = "C:/Users/salva/PycharmProjects/Data/aibedo/xarrays"

ds_mlp = xr.load_dataset(f"{PDIR}/MLP-no-constraints-37z9auhs.nc")
ds_sunet = xr.load_dataset(f"{PDIR}/SUNet-no-constraints-2frxrgnq.nc")
ds_all_constraints_mlp = xr.load_dataset(f"{PDIR}/MLP-with-constraints-2ymsmghz.nc")

kwargs = dict(ds=ds_mlp.isel(snapshot=slice(-6, -1)), x='longitude', y='latitude', col='snapshot')
xr.plot.scatter(hue='pr_targets', **kwargs)
xr.plot.scatter(hue='tas_targets', **kwargs)
xr.plot.scatter(hue='psl_targets', **kwargs)
vars_to_plot = 'all'
vars_to_plot = ['tas']
ds = ds_mlp
vars_to_plot = ds_mlp.variable_names.split(";") if vars_to_plot == "all" else vars_to_plot
for var in vars_to_plot:
    var_id = var.replace('_targets', '').replace('_preds', '')
    print(f"Plotting {var_id}", var)
    ds_all_constraints_mlp[var_id + '_targets'] = ds_mlp[var_id + '_targets']
    ani1 = animate_snapshots(ds, var_to_plot=var_id + '_targets')
    ani2 = animate_snapshots(ds, var_to_plot=var_id + '_preds')
plt.show()

ani = animate_snapshots(ds_all_constraints_mlp, var_to_plot='tas_preds')
plt.show()

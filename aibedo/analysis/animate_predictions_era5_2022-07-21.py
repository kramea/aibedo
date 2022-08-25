import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import animation

from aibedo.utilities.plotting import animate_snapshots

# ---------------> Edit these directory pointers!
PDIR = "C:/Users/salva/PycharmProjects/Data/aibedo/xarrays"
dir_to_save = "C:/Users/salva/PycharmProjects/Data/aibedo/animations"

# ---------------> Edit these xarray filenames to the one(s) you want to plot!
ds_mlp = xr.load_dataset(f"{PDIR}/MLP-no-constraints-37z9auhs.nc")
ds_sunet = xr.load_dataset(f"{PDIR}/SUNet-no-constraints-2frxrgnq.nc")
ds_all_constraints_mlp = xr.load_dataset(f"{PDIR}/MLP-with-constraints-2ymsmghz.nc")
ds_all_constraints_mlp2 = xr.load_dataset(f"{PDIR}/34q3echu.nc")
ds_all_constraints_mlp2.attrs['id'] = '34q3echu'

# Select which of the above datasets to plot:
ds = ds_all_constraints_mlp2

# Select which variable to plot:
vars_to_plot = 'all'  # alternative: ['tas']

vars_to_plot = ds.variable_names.split(";") if vars_to_plot == "all" else [vars_to_plot]
# Show GIFs by target variable, 1by1
kwargs = dict(num_snapshots=10, interval=3000)
GIF = True  # False does not seem to work
dpi = 80
ending = '.gif' if GIF else '.mp4'
for var in vars_to_plot:
    var_id = var.replace('_targets', '').replace('_preds', '')
    if var_id + '_targets' not in ds.variables:
        # in case the targets were not saved to the xarray, we need to add them from another xarray
        ds[var_id + '_targets'] = ds_mlp[var_id + '_targets']
    print(f"Plotting {var_id}", var)
    # kwargs['vmin'], kwargs['vmax'] = ds[var_id + '_targets'].min(), ds[var_id + '_targets'].max()

    # Animate targets
    ani1, scatter_fig1 = animate_snapshots(ds, var_to_plot=var_id + '_targets', **kwargs)
    writer = animation.PillowWriter(fps=4) if GIF else animation.FFMpegWriter(fps=60)
    ani1.save(f'{dir_to_save}/ERA5_targets_{var_id}.{ending}', dpi=dpi, writer=writer)

    cbar = scatter_fig1.colorbar
    kwargs2 = {**kwargs, 'cmap': scatter_fig1.cmap.name, 'vmin': cbar.vmin, 'vmax': cbar.vmax}

    # Animate predictions
    writer = animation.PillowWriter(fps=4) if GIF else animation.FFMpegWriter(fps=60)
    ani2, _ = animate_snapshots(ds, var_to_plot=var_id + '_preds', **kwargs2)
    ani2.save(f'{dir_to_save}/ERA5_preds_{ds.id}_{var_id}.{ending}', dpi=dpi, writer=writer)

    # plt.show()

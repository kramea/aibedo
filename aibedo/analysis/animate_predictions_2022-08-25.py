import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import animation

from aibedo.utilities.plotting import animate_snapshots

# ---------------> Edit these directory pointers!
PDIR = "C:/Users/salva/PycharmProjects/Data/aibedo/xarrays"
dir_to_save = "C:/Users/salva/PycharmProjects/Data/aibedo/animations"

# ---------------> Edit these xarray filenames to the one(s) you want to plot!
ds_mlp1 = xr.load_dataset(f"{PDIR}/predictions_CESM2_val_3b6fg6sb_MLP_spherical_0constraints.nc")
ds_mlp2 = xr.load_dataset(f"{PDIR}/predictions_CESM2_val_ubjfpr0f_MLP_spherical_4constraints.nc")
ds_fno = xr.load_dataset(f"{PDIR}/predictions_CESM2_val_3lax8md0_FNO_spherical.nc")
ds_mlp1.attrs['id'] = '3b6fg6sb'
ds_mlp2.attrs['id'] = 'ubjfpr0f'
ds_fno.attrs['id'] = '3lax8md0'

# -------------> EDIT THIS: select which of the above datasets to plot:
ds_to_animate = ds_mlp1

# Select which variable to plot:
vars_to_plot = 'all'  # alternative: ['tas'], ['tas', 'pr'], etc.

vars_to_plot = ds_to_animate.variable_names.split(";") if vars_to_plot == "all" else vars_to_plot
# Show GIFs by target variable, 1by1
kwargs = dict(num_snapshots=10, interval=3000)
GIF = True  # False does not seem to work
dpi = 80
ending = '.gif' if GIF else '.mp4'
dataset_name = ds_to_animate.attrs['dataset_name']
for var in vars_to_plot:
    var_id = var.replace('_targets', '').replace('_preds', '')

    print(f"Plotting {var_id}", var)

    # Animate targets
    ani1, scatter_fig1 = animate_snapshots(ds_to_animate, var_to_plot=var_id + '_targets', **kwargs)
    writer = animation.PillowWriter(fps=4) if GIF else animation.FFMpegWriter(fps=60)
    ani1.save(f'{dir_to_save}/{dataset_name}_targets_{var_id}.{ending}', dpi=dpi, writer=writer)

    cbar = scatter_fig1.colorbar
    kwargs_preds = {**kwargs, 'cmap': scatter_fig1.cmap.name, 'vmin': cbar.vmin, 'vmax': cbar.vmax}

    # Animate predictions
    writer = animation.PillowWriter(fps=4) if GIF else animation.FFMpegWriter(fps=60)
    ani2, _ = animate_snapshots(ds_to_animate, var_to_plot=var_id + '_preds', **kwargs_preds)
    ani2.save(f'{dir_to_save}/{dataset_name}_preds_{ds_to_animate.id}_{var_id}.{ending}', dpi=dpi, writer=writer)

    # plt.show()

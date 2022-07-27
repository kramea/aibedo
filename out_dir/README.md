# *Shared drive*
Can be found at [this link](https://parc-my.sharepoint.com/:f:/p/pmitra/EgiutXMPCMtPsVHyY_Hvsw0BNZtnOK2rCP80EnM0mrGzOg?e=sLePgj).

## Saved models and predictions

In the shared drive above, several predictions (and targets) have been saved for various models.
Unless otherwise noted, all models use AdamW as optimizer with 2e-4 as learning rate and 
an exponential scheduler(gamma=0.98). They are trained for 25 epochs, and the best model is 
saved based on the validation MSE loss ('val/mse'). This is the same model used for inference, 
and thus for generating the predictions saved in the drive.

Unless otherwise noted:
- MLP: Four layers of 1024 neurons each

**Legend:**
- [34q3echu.nc](https://parc-my.sharepoint.com/:u:/p/pmitra/EWUgWktVj71CslZc-JTThb4BX07vou_CO2CLYaQw-ZIHBw?e=Rwy8vD)
 is a MLP predicting 'tas', 'pr', 'ps' with physics constraints as per 
 [0, 0, 1000, 1.0, 0.0001] (without updates on the scaling of constraint #5) for CESM2 validation set.
- [37tl5tsy.nc](https://parc-my.sharepoint.com/:u:/p/pmitra/EZh1kq_uAGxNjegYJhUyA3QBtIkMn_gdL7mRQhms9qqvig?e=0aLnO8)
 is the analogous MLP to 34q3echu.nc, but without using any constraints (all coefficients are 0)
- val_1ul6rx81.nc analogous to above, but using physics constraints as per [0, 0.005, 1000, 1, 1], i.e. including constraint #2

## Getting started with the saved predictions

To use and analyze the saved predictions, you need to download the desired file from the shared drive
(a file ending with .nc) and load the file as an xarray Dataset as below:

```python
import xarray as xr
ds = xr.open_dataset('<filename-from-shared-drive>.nc')
```

### Plotting the predictions/targets

Given a xarray dataset as above, you can plot the predictions/targets/errors in various ways:

##### Plotting the predictions for multiple snapshots

Please see the code snippet below or many of the notebooks in the [notebooks/](../notebooks/) directory, 
such as [this one](../notebooks/2022-07-26-ps-pr-tas-val-set-CESM2.ipynb).

```python

from aibedo.utilities.plotting import data_snapshots_plotting

ds = ... # xarray dataset
vars_to_plot = ['pr']  # can be any list of the variables in the dataset
_ = data_snapshots_plotting(
 ds, 
 title=" -- my model",
 plot_error=False,            # Whether to plot the error too in the third row (e.g. preds - targets)
 cmap=None,                   # can be any matplotlib colormap
 vars_to_plot=vars_to_plot
)
```

##### Animate the predicted and targeted values for multiple snapshots in a video or GIF 

Please see the code-block below, and 
[this script](../analysis/animate_predictions_era5_2022-07-21.py) for a full-fledged example.

```python
from matplotlib import animation
from aibedo.utilities.plotting import animate_snapshots

ds = ... # xarray dataset
gif_kwargs = dict(num_snapshots=12, interval=1000)  #  num_snapshots: number of snapshots to animate 
ani_targets, _ = animate_snapshots(ds, var_to_plot='pr_targets', **gif_kwargs)
ani_preds, _ = animate_snapshots(ds, var_to_plot='pr_preds', **gif_kwargs)
    
# Save the animation as a GIF
ani_targets.save(f'<where-to-save-the-animation>_targets.gif', dpi=80, writer=animation.PillowWriter(fps=2))
ani_preds.save(f'<where-to-save-the-animation>_preds.gif', dpi=80, writer=animation.PillowWriter(fps=2))

```

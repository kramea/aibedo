import os.path
import random
from typing import List, Union, Sequence, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import einops
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
from matplotlib import animation
import matplotlib.cm as cm

from aibedo.constants import CLIMATE_MODELS_ALL
from aibedo.utilities.naming import var_names_to_clean_name
from aibedo.utilities.utils import raise_error_if_invalid_value
from aibedo.utilities.wandb_api import groupby

var_names_to_cmap = {
    'tas_pre': 'coolwarm',  # rainbow
    'psl_pre': 'Spectral',
    'ps_pre': 'Spectral',
    'pr_pre': 'bwr',
    'tas': 'coolwarm',  # rainbow
    'psl': 'Spectral',
    'ps': 'Spectral',
    'pr': 'bwr',
}

ESM_to_color = {esm: cm.Spectral(np.linspace(0, 1, len(CLIMATE_MODELS_ALL)))[i]  # rainbow  # Spectral
                for i, esm in enumerate(CLIMATE_MODELS_ALL)}


def set_labels_and_ticks(ax,
                         title: str = "",
                         xlabel: str = "", ylabel: str = "",
                         xlabel_fontsize: int = 10, ylabel_fontsize: int = 14,
                         xlim=None, ylim=None,
                         xticks=None, yticks=None,
                         xtick_labels=None, ytick_labels=None,
                         x_ticks_rotation=None, y_ticks_rotation=None,
                         xticks_fontsize: int = None, yticks_fontsize: int = None,
                         title_fontsize: int = None,
                         logscale_y: bool = False,
                         show: bool = True,
                         grid: bool = True,
                         legend: bool = True, legend_loc='best', legend_prop=10,
                         full_screen: bool = False,
                         tight_layout: bool = True,
                         save_to: str = None
                         ):
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if xtick_labels is not None:
        ax.set_xticklabels(xtick_labels, rotation=x_ticks_rotation)
    if xticks_fontsize:
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(xticks_fontsize)
        # tick.label.set_rotation('vertical')

    if logscale_y:
        ax.set_yscale('log')
    if yticks is not None:
        ax.set_yticks(yticks)
    if ytick_labels is not None:
        ax.set_yticklabels(ytick_labels, rotation=y_ticks_rotation)
    if yticks_fontsize:
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(yticks_fontsize)

    if grid:
        ax.grid()
    if legend:
        ax.legend(loc=legend_loc, prop={'size': legend_prop})  # if full_screen else ax.legend(loc=legend_loc)

    if tight_layout:
        plt.tight_layout()

    save_figure(save_to, full_screen=full_screen)

    if show:
        plt.show()


def adjust_subplots_spacing(
        left=0.125,  # the left side of the subplots of the figure
        right=0.9,  # the right side of the subplots of the figure
        bottom=0.1,  # the bottom of the subplots of the figure
        top=0.9,  # the top of the subplots of the figure
        wspace=0.2,  # the amount of width reserved for blank space between subplots
        hspace=0.2,  # the amount of height reserved for white space between subplots
):
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


def save_figure(save_to: str, full_screen: bool = False, bbox_inches='tight'):
    if save_to is not None:
        if full_screen:
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
            plt.savefig(save_to, bbox_inches=bbox_inches)
            manager.full_screen_toggle()
        else:
            plt.savefig(save_to, bbox_inches=bbox_inches)


def get_vars_to_plot(vars_to_plot: Union[str, List[str]], postprocessed_xarray: xr.Dataset) -> List[str]:
    """
    Syntactic sugar to get the output variables to plot, e.g. by passing 'all', 'denormalized', 'normalized', etc.
    """
    if isinstance(vars_to_plot, str):
        raise_error_if_invalid_value(vars_to_plot, ['all', 'denormalized', 'normalized', 'raw'], 'vars_to_plot')
    if vars_to_plot == "all":
        output_vars = postprocessed_xarray.variable_names.split(";")
    elif vars_to_plot == "normalized":
        output_vars = [x for x in postprocessed_xarray.variable_names.split(";") if '_pre' in x]
    elif vars_to_plot in ["denormalized", 'raw']:
        output_vars = [x for x in postprocessed_xarray.variable_names.split(";") if '_pre' not in x]
    else:
        assert isinstance(vars_to_plot, list), f"vars_to_plot must be a list, but is {type(vars_to_plot)}"
        output_vars = vars_to_plot
    return output_vars


def data_mean_plotting(postprocessed_xarray: xr.Dataset,
                       error_to_plot: str = "mae",
                       vars_to_plot: Union[str, List[str]] = 'all',
                       data_dim: str = 'snapshot',
                       longitude_dim: str = 'longitude',
                       latitude_dim: str = 'latitude',
                       robust: bool = True,
                       plot_only_errors: bool = False,
                       marker_size: int = 2,
                       title_fontsize: int = 24,
                       **kwargs
                       ):
    """
    Plot the mean of the data along the data_dim (error, and optionally targets/ground-truth and predictions).
    Args:
        postprocessed_xarray: The xarray Dataset with the data to plot.
        error_to_plot (str): Which error to plot. Default is 'mae'. Other options: 'bias', 'mae_score'
        data_dim (str): the data dimension (i.e. the example/time dimension)
        longitude_dim (str): name of the longitude dimension
        latitude_dim (str): name of the latitude dimension
        robust: If True: visualize the data without outliers.
                        This will use the 2nd and 98th percentiles of the data to compute the color limits.
        plot_only_errors: Whether to only plot the errors or also the targets and predictions (for reference).
        marker_size: The size of the markers in the plot.
        title_fontsize: The fontsize of the title.

    Returns: A tuple of the figure and the axes
    """
    ds_data_mean = postprocessed_xarray.mean(dim=data_dim)
    proj = ccrs.PlateCarree()
    kwargs = dict(
        x=longitude_dim,
        y=latitude_dim,
        transform=proj,
        subplot_kws={'projection': proj},
        s=marker_size,  # marker size
        robust=robust,
        cbar_kwargs={'shrink': 0.8,  # make cbar smaller/larger
                     'pad': 0.01,  # padding between right-ost subplot and cbar
                     'fraction': 0.05},
        **kwargs
    )

    output_vars = get_vars_to_plot(vars_to_plot, postprocessed_xarray)

    nrows = 1 if plot_only_errors else 3
    ncols = len(output_vars)
    fig, axs = plt.subplots(nrows, ncols,
                            subplot_kw={'projection': proj},
                            gridspec_kw={'wspace': 0.07, 'hspace': 0,
                                         'top': 1., 'bottom': 0., 'left': 0., 'right': 1.},
                            figsize=(ncols * 12, nrows * 6)  # <- adjust figsize but keep ratio ncols/nrows
                            )

    for i, var in enumerate(output_vars):
        bulk_err = float(getattr(postprocessed_xarray, f'{var}_{error_to_plot}').mean().data)
        top_most_var_ax = axs[i] if plot_only_errors else axs[0, i]
        error_ax = top_most_var_ax if plot_only_errors else axs[2, i]

        if not plot_only_errors:
            for row, dtype in enumerate(["targets", 'preds']):
                pc = xr.plot.scatter(ds_data_mean, hue=f'{var}_{dtype}', cmap=var_names_to_cmap[var], ax=axs[row, i],
                                     **kwargs)
                # Edit the colorbar
                pc.colorbar.set_label(f"{var} {dtype.capitalize()}", size=20)  # , weight='bold')
                pc.colorbar.ax.tick_params(labelsize=15)

        # Plot the error
        pc = xr.plot.scatter(ds_data_mean, hue=f'{var}_{error_to_plot}', ax=error_ax, **kwargs)
        pc.colorbar.set_label(f"{var} {error_to_plot.upper()}", size=20)  # , weight='bold')
        pc.colorbar.ax.tick_params(labelsize=15)

        # Set column title
        if not plot_only_errors:
            title_v = f"{var_names_to_clean_name()[var]}"
            top_most_var_ax.set_title(title_v, fontsize=title_fontsize)
        bulk_err_f = f"{bulk_err:.6f}" if var == 'pr' else f"{bulk_err:.3f}"
        error_ax.set_title(f"${error_to_plot.upper()}={bulk_err_f}$", fontsize=title_fontsize - 2)
    for ax in np.array(axs).flatten():
        ax.coastlines()
    return fig, axs


def data_snapshots_plotting(postprocessed_xarray: xr.Dataset,
                            error_to_plot: str = "bias",
                            vars_to_plot: List[str] = 'all',
                            snapshots_to_plot: List[int] = None,
                            num_snapshots_to_plot: int = 5,
                            data_dim: str = 'snapshot',
                            longitude_dim: str = 'longitude',
                            latitude_dim: str = 'latitude',
                            robust: bool = True,
                            same_colorbar_for_preds_and_targets: bool = True,
                            plot_only_preds: bool = False,
                            plot_error: bool = True,
                            data_to_plot=('targets', 'preds', 'error'),
                            marker_size: int = 2,
                            coastlines_linewidth: float = 1,
                            title: str = "",
                            title_fontsize: int = 18,
                            cmap='auto',
                            seed=7,
                            **kwargs
                            ):
    """
    This function will plot the predictions and/or targets and/or global errors for multiple snapshots/time-steps.

    Args:
        postprocessed_xarray: The xarray Dataset with the data to plot.
        error_to_plot (str): Which error to plot. Default is 'mae'. Other options: 'bias', 'mae_score'
        vars_to_plot: Which output variables to plot. Default is 'all' (plot all output variables).
        snapshots_to_plot: A list of snapshots to plot. If None, then sample a random subset of the snapshots.
        num_snapshots_to_plot (int): The number of snapshots to plot (subsamples from the time dimension).
        data_dim (str): the data dimension (i.e. the example/time dimension)
        longitude_dim (str): name of the longitude dimension
        latitude_dim (str): name of the latitude dimension
        robust:
            If True: visualize the data without outliers.
                    This will use the 2nd and 98th percentiles of the data to compute the color limits.
        same_colorbar_for_preds_and_targets:
            If True: use the same colorbar (magnitude) is used for the predictions and targets subplots.
                    This can ease the visual comparison of the predictions and targets.
        plot_only_preds (bool): If True, only plot the predictions, else plot targets, preds, and optionally the error.
        plot_error (bool): If True, plot the error. Default is True.
        marker_size: The size of the markers in the plot.
        coastlines_linewidth (float): The width of the coastlines/continent borders.
        title (str): The title of the plot (appended to the variable name title if only_plot_preds is False).
                        If None, then no title is shown.
        title_fontsize: The fontsize of the title.
        cmap (str): The colormap to use. Default is 'auto' (== pre-defined cmaps), use None for matplotlib default.
        seed (int): The seed for the random sampling of the snapshots.

    Returns:
        The matplotlib PathCollection, pcs, as a dictionary with one key for each output variable name.
        Each value pcs[output_var_name] is a dict that has the following keys:
            - 'targets': the matplotlib PathCollection for the targets of the output variable
            - 'preds': the matplotlib PathCollection for the predictions of the output variable
            - error_to_plot: the matplotlib PathCollection for the error of the output variable
    """
    if snapshots_to_plot is None:
        random.seed(seed)
        # Sample a random subset of the snapshots
        snaps = sorted(random.sample(range(postprocessed_xarray.dims[data_dim]), num_snapshots_to_plot))
    else:
        snaps = snapshots_to_plot

    output_vars = get_vars_to_plot(vars_to_plot, postprocessed_xarray)

    data_to_plot = list(set(data_to_plot))
    assert all(x in ['targets', 'preds', 'error'] for x in data_to_plot)
    assert len(data_to_plot) > 0
    plot_only_preds = plot_only_preds or ['preds'] == data_to_plot
    if plot_only_preds:
        data_to_plot = ['preds']
    if len(data_to_plot) == 3 and not plot_error:  # only with default data_to_plot arg
        data_to_plot.remove('error') if 'error' in data_to_plot else None
    if 'error' in data_to_plot and f'{vars_to_plot[0]}_{error_to_plot}' not in postprocessed_xarray.keys():
        raise ValueError(f"The error to plot {vars_to_plot[0]}_{error_to_plot} is not in the dataset! ")

    proj = ccrs.PlateCarree()
    label_fontsize = title_fontsize // 1.5
    ds_snaps = postprocessed_xarray.isel({'snapshot': snaps})
    kwargs2 = dict(
        ds=ds_snaps,
        x=longitude_dim,
        y=latitude_dim,
        col=data_dim,
        transform=proj,
        subplot_kws={'projection': proj},
        s=marker_size,  # marker size
        robust=robust,
        cbar_kwargs={'shrink': 0.5,  # make cbar smaller/larger
                     'pad': 0.01,  # padding between right-ost subplot and cbar
                     'fraction': 0.05}
    )
    p_target = p_err = p_pred = None
    ps = dict()
    for var in output_vars:
        cmap = p_cmap = var_names_to_cmap[var] if cmap == 'auto' else cmap
        if 'targets' in data_to_plot:
            p_target = xr.plot.scatter(hue=f'{var}_targets', cmap=cmap, **kwargs, **kwargs2)
            if same_colorbar_for_preds_and_targets:
                # Set the same colorbar for the predictions and targets subplots
                kwargs['vmin'], kwargs['vmax'] = p_target.cbar.vmin, p_target.cbar.vmax
                p_cmap = p_target.cbar.cmap
            # Set coastlines of all axes
            for ax in list(p_target.axes.flat):
                ax.coastlines(linewidth=coastlines_linewidth)
        if 'preds' in data_to_plot:
            p_pred = xr.plot.scatter(hue=f'{var}_preds', cmap=p_cmap, **kwargs, **kwargs2)
            # Set coastlines of all axes
            for ax in list(p_pred.axes.flat):
                ax.coastlines(linewidth=coastlines_linewidth)

        if 'error' in data_to_plot:
            kwargs_error = kwargs if 'targets' not in data_to_plot and 'preds' not in data_to_plot else dict()
            p_err = xr.plot.scatter(hue=f'{var}_{error_to_plot}', **kwargs_error, **kwargs2)
            for ax in list(p_err.axes.flat):
                ax.coastlines(linewidth=coastlines_linewidth)
        ps[var] = {'targets': p_target, 'preds': p_pred, error_to_plot: p_err}

        # Set row names (ylabel) for the leftmost subplot of each row
        for dtype in data_to_plot:
            k = error_to_plot if dtype == 'error' else dtype
            pc = ps[var][k]
            ylabel = error_to_plot.upper() if dtype == 'error' else dtype.capitalize()
            # pc.axes.flat[0].set_ylabel(ylabel, size=label_fontsize)
            # Edit the colorbar
            pc.cbar.set_label(ylabel, size=label_fontsize)
            pc.cbar.ax.tick_params(labelsize=label_fontsize // 1.7)

        if plot_only_preds:
            p_pred.fig.suptitle(title, fontsize=title_fontsize, y=0.9)
        elif title is not None and 'targets' in data_to_plot:
            # Add title to variable subplots at the middle top
            title_v = f"{var_names_to_clean_name()[var]} {title}"  # (${error_to_plot.upper()}={snapshot_err:.3f}$)"
            p_target.fig.suptitle(title_v, fontsize=title_fontsize, y=0.9)

        if 'targets' in data_to_plot and 'preds' in data_to_plot:
            # remove snapshot = snapshot_id title from middle plots (already included in targets (top row))
            for ax in list(p_pred.axes.flat):
                ax.set_title('')

        if 'error' in data_to_plot:
            # Add bulk errors (per snapshot) as title to error subplots
            snapshot_mae_f = snapshot_bias_f = ""
            for i, (ax, snap_num) in enumerate(zip(p_err.axes.flat, p_err.col_names)):
                if hasattr(p_err.data, f'{var}_mae'):
                    snapshot_mae = float(getattr(p_err.data, f'{var}_mae').sel({data_dim: snap_num}).mean().data)
                    snapshot_mae_f = f"$MAE={snapshot_mae:.6f}$" if var == 'pr' else f"$MAE={snapshot_mae:.3f}$"

                if hasattr(p_err.data, f'{var}_bias'):
                    snapshot_bias = float(getattr(p_err.data, f'{var}_bias').sel({data_dim: snap_num}).mean().data)
                    snapshot_bias_f = f"Bias$={snapshot_bias:.7f}$" if var == 'pr' else f"Bias$={snapshot_bias:.3f}$"

                ax.set_title(
                    f"{snapshot_mae_f} {snapshot_bias_f}",
                    fontsize=label_fontsize,
                )

    return ps


def animate_snapshots(postprocessed_xarray: xr.Dataset,
                      var_to_plot: str = 'pr_preds',
                      num_snapshots: int = 12,
                      cmap=None,
                      interval=400,
                      **kwargs
                      ):
    pa_kwargs = dict(x='longitude', y='latitude', cbar_kwargs={'shrink': 0.5, 'pad': 0.01, 'fraction': 0.03})
    plt.rc('animation', html='jshtml')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    n_frames = len(postprocessed_xarray.indexes['snapshot'])
    first_snap = n_frames - num_snapshots
    first_snap_xr = postprocessed_xarray.isel(snapshot=first_snap)
    scatter = xr.plot.scatter(first_snap_xr, hue=var_to_plot, cmap=cmap, ax=ax, **pa_kwargs, **kwargs)

    frames = range(first_snap, n_frames)
    xr_preds = getattr(postprocessed_xarray, var_to_plot)  # dont do (yet): .isel(snapshot=frames)

    ax.gridlines(alpha=0.5)
    ax.coastlines(resolution="50m", color="white")

    def animate_preds(i):
        scatter.set_array(xr_preds.isel(snapshot=i).values)  # update animation
        ax.set_title(f'Snapshot = {i}')

    return animation.FuncAnimation(fig, animate_preds, frames=frames, interval=interval, blit=False), scatter


def add_grouped_latitude(ds, resolution: float = 1.0):
    lats_grouped = np.array([round(x) for x in ds.latitude.values])
    ds = ds.assign_coords(latitude_grouped=('spatial_dim', lats_grouped))
    return ds


def get_labels_and_styles_for_lines(objects_to_plot: List[Any], labels: List[str] = None, linestyles: List[str] = None):
    if not isinstance(objects_to_plot, list):
        objects_to_plot = [objects_to_plot]
    assert len(objects_to_plot) > 0, "No postprocessed xarrays provided to plot!"
    if labels is None:
        labels = [""] * len(objects_to_plot)
    if linestyles is None:
        linestyles = ["-"] if len(objects_to_plot) < 2 else ['--', ':', '-.'] * len(objects_to_plot)
    assert len(labels) == len(objects_to_plot), "Number of labels must match number of objects_to_plot!"
    return objects_to_plot, labels, linestyles


def zonal_plotting(postprocessed_xarrays: xr.Dataset,
                   labels: List[str] = None,
                   linestyles=None,
                   vars_to_plot: List[str] = 'all',
                   data_dim: str = 'snapshot',
                   latitude_dim: str = 'latitude_grouped',
                   axes: List[plt.Axes] = None,
                   fontsize=20,
                   **kwargs
                   ):
    postprocessed_xarrays, labels, linestyles = get_labels_and_styles_for_lines(postprocessed_xarrays, labels, linestyles)
    output_vars = get_vars_to_plot(vars_to_plot, postprocessed_xarrays[0])
    nrows, ncols = (1, 3) if len(output_vars) <= 3 else (2, 3) if len(output_vars) <= 6 else (3, 3)
    fig, axs = plt.subplots(nrows, ncols) if axes is None else (None, axes)
    axs = axs.flatten() if nrows > 1 else axs
    dset_name = postprocessed_xarrays[0].attrs['dataset_name']
    assert all([pds.attrs['dataset_name'] == dset_name for pds in postprocessed_xarrays]), "All datasets must be from the same dataset!"

    for i, (pds, label, ls) in enumerate(zip(postprocessed_xarrays, labels, linestyles)):
        if latitude_dim == 'latitude_grouped':
            pds = add_grouped_latitude(pds)
        zonal_mean = pds.groupby(latitude_dim).mean().mean(dim=data_dim)

        for j, var in enumerate(output_vars):
            zonal_var_preds = getattr(zonal_mean, f"{var}_preds")
            zonal_var_preds.plot(x=latitude_dim, ax=axs[j], label=f'Preds {label}', linestyle=ls, **kwargs)
            if i == 0:
                # Plot the targets/groundtruth if this is the first plot
                zonal_var_targets = getattr(zonal_mean, f"{var}_targets")
                ls = '-' if axes is None else ':'  # if plotting on top of another plot, use different linestyle
                zonal_var_targets.plot(x=latitude_dim, ax=axs[j], label=f'Targets {dset_name}', color='k', linestyle=ls, **kwargs)

            axs[j].set_ylabel(f"{var.upper()}", fontsize=fontsize)

    for ax in axs:
        ax.legend(prop=dict(size=fontsize + 3))
        ax.grid()
        ax.set_xlabel('Latitude', fontsize=fontsize)
    # set title for all axes
    plt.suptitle(f"{dset_name}", fontsize=fontsize)

    return fig, axs


def zonal_error_plotting(postprocessed_xarrays: List[xr.Dataset],
                         labels: List[str] = None,
                         linestyles=None,
                         error_to_plot: str = "bias",
                         vars_to_plot: List[str] = 'all',
                         data_dim: str = 'snapshot',
                         longitude_dim: str = 'longitude',
                         latitude_dim: str = 'latitude_grouped',
                         fontsize=20,
                         **kwargs
                         ):
    postprocessed_xarrays, labels, linestyles = get_labels_and_styles_for_lines(postprocessed_xarrays, labels, linestyles)
    output_vars = get_vars_to_plot(vars_to_plot, postprocessed_xarrays[0])
    nrows = 1  # if plot_only_errors else 3
    ncols = len(output_vars)
    fig, axs = plt.subplots(nrows, ncols)
    axs = axs.flatten() if nrows > 1 else axs
    for pds, label, ls in zip(postprocessed_xarrays, labels, linestyles):
        pds = add_grouped_latitude(pds)
        zonal_err = pds.groupby(latitude_dim).mean().mean(dim=data_dim)
        # same as: pds.mean(dim=data_dim).groupby(latitude_dim).mean()
        for j, var in enumerate(output_vars):
            err_id = f"{var}_{error_to_plot}"
            # varTmean = pds.groupby(var.time.dt.month).mean(dim='time')
            zonal_err_var = getattr(zonal_err, err_id)
            zonal_err_var.plot(x=latitude_dim, ax=axs[j], label=label, linestyle=ls, **kwargs)
            axs[j].set_ylabel(f"{var.upper()} {error_to_plot}", fontsize=fontsize)
    for ax in axs:
        ax.legend(prop=dict(size=fontsize + 3))
        ax.grid()
        ax.set_xlabel('Latitude', fontsize=fontsize)

    return fig, axs


def plot_training_set_vs_test_performance(
        runs_to_use_df: pd.DataFrame,
        metrics_to_plot: Sequence[str] = ('val/mse_epoch', 'test/MERRA2/mse_epoch', 'test/ERA5/mse_epoch'),
        keep_columns: Sequence[str] = ('id',),
        max_error_to_plot: float = None,
        order_by_performance: bool = True,
        save_to_dir: str = None,
        **kwargs
):
    """
    Plot the influence of the used training set (ESM input) on the test performance (a reanalysis dataset).
    """
    grouped_runs_stats = groupby(runs_to_use_df,
                                 group_by=['datamodule/input_filename'],
                                 keep_columns=list(keep_columns),
                                 metrics=list(metrics_to_plot))

    title = kwargs.pop('title', None)
    n_metrics = len(metrics_to_plot)
    fig, axs = plt.subplots(n_metrics, 1)
    for i, metric in enumerate(metrics_to_plot):
        ax = axs[i]
        xlabels, errors, stds = [], [], []
        for j, run_group in grouped_runs_stats.iterrows():
            run_ESM = run_group['datamodule/input_filename'].split('.')[2]
            y = run_group[f"{metric}/mean"]
            y_std = run_group[f"{metric}/std"]
            first_run_id = run_group['id']
            if y != y or y_std != y_std:  # NaN
                continue

            if max_error_to_plot and y > max_error_to_plot:
                print(f"run {run_group} has bad {metric} score={y}, skipping it")
                continue
            print(f"run {run_group} has {metric} score={y}")
            errors.append(y)
            stds.append(y_std)
            xlabels += [run_ESM]

        colors = [ESM_to_color[esm] for esm in xlabels]  # cm.rainbow(np.linspace(0, 1, len(xlabels)))
        if order_by_performance:
            # order positions by performance (lowest to highest)
            errors, stds, xlabels, colors = map(list,
                                                (zip(*sorted(zip(errors, stds, xlabels, colors), key=lambda x: x[0]))))

        x_pos = np.arange((len(xlabels)))
        p = ax.bar(x_pos, errors, yerr=stds, color=colors, align='center', alpha=0.5, ecolor='black', capsize=10)
        max_diff = max(errors) - min(errors)
        ax.set_ylim([max(min(errors) - max_diff - max(stds), ax.get_ylim()[0]), ax.get_ylim()[1]])
        metric_name = metric.upper().replace('_EPOCH', '').replace('TEST/', '').replace('/', ' ').replace('VAL', 'Val')
        for ref in ["ERA5", "MERRA2"]:
            if ref in title and ref in metric_name:
                metric_name = metric_name.replace(ref, "").strip()

        title_save = f"{title}_" if title else ""
        save_to = os.path.join(save_to_dir, f'{title_save}{metric_name}.png') if save_to_dir else None
        if i < len(metrics_to_plot) - 1:
            pass  # xlabels = []
        set_labels_and_ticks(
            ax,
            xlabel='', ylabel=metric_name,
            xticks=x_pos, xtick_labels=xlabels,
            xlabel_fontsize=14 if n_metrics <= 2 else 6,
            ylabel_fontsize=14 if n_metrics <= 2 else 8,
            xticks_fontsize=6,
            yticks_fontsize=14 if n_metrics <= 2 else 9,
            x_ticks_rotation=10,  # 45
            show=False, legend=False, legend_loc='best',
            grid=True,
            save_to=save_to if save_to and i == len(metrics_to_plot) - 1 else None,
            title=title if i == 0 else None,
            **kwargs
        )
    plt.subplots_adjust(wspace=0.01, hspace=0.26, bottom=0.06, left=0.055, right=0.99, top=0.96)
    return fig, axs

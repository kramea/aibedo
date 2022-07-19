import random
from typing import List, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import einops
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs

from aibedo.utilities.naming import var_names_to_clean_name

var_names_to_cmap = {
    'tas_pre': 'coolwarm',  # rainbow
    'psl_pre': 'Spectral',
    'pr_pre': 'bwr',
    'tas': 'coolwarm',  # rainbow
    'psl': 'Spectral',
    'pr': 'bwr',
}


def get_subplots(nrows: int = 1,
                 ncols: int = 1,
                 xlabel: str = None,
                 ylabel: str = None,
                 sharex: bool = False,
                 sharey: bool = False,
                 flatten_axes: bool = True,
                 **kwargs):
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, sharex=sharex, sharey=sharey, **kwargs)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    if xlabel:
        if sharex:
            plt.setp(axs[-1, :], xlabel=xlabel)
        else:
            plt.xlabel(xlabel)
    if ylabel:
        if sharey:
            plt.setp(axs[:, 0], ylabel=ylabel)
        else:
            plt.ylabel(ylabel)

    if flatten_axes:
        axs = np.array(axs).flatten()  # flatten potentially 2d axes array into 1d
    return fig, axs


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


def save_figure(save_to: str, full_screen: bool = False, bbox_inches='tight'):
    if save_to is not None:
        if full_screen:
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
            plt.savefig(save_to, bbox_inches=bbox_inches)
            manager.full_screen_toggle()
        else:
            plt.savefig(save_to, bbox_inches=bbox_inches)


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
                     'fraction': 0.05}
    )
    if vars_to_plot == "all":
        output_vars = postprocessed_xarray.variable_names.split(";")
    elif vars_to_plot == "normalized":
        output_vars = [x for x in postprocessed_xarray.variable_names.split(";") if '_pre' in x]
    elif vars_to_plot == "denormalized":
        output_vars = [x for x in postprocessed_xarray.variable_names.split(";") if '_pre' not in x]
    else:
        assert isinstance(vars_to_plot, list), f"vars_to_plot must be a list, but is {type(vars_to_plot)}"
        output_vars = vars_to_plot

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
        bulk_err_f = f"{bulk_err:.6f}" if var == 'pr' else f"{bulk_err:.3f}"
        title_v = f"{var_names_to_clean_name()[var]} (${error_to_plot.upper()}={bulk_err_f}$)"
        top_most_var_ax.set_title(title_v, fontsize=title_fontsize)
    for ax in np.array(axs).flatten():
        ax.coastlines()
    return fig, axs


def data_snapshots_plotting2(postprocessed_xarray: xr.Dataset,
                             error_to_plot: str = "mae",
                             num_snapshots_to_plot: int = 5,
                             data_dim: str = 'snapshot',
                             longitude_dim: str = 'longitude',
                             latitude_dim: str = 'latitude',
                             robust: bool = True,
                             same_colorbar_for_preds_and_targets: bool = True,
                             marker_size: int = 2,
                             title_fontsize: int = 18,
                             ):
    """

    Args:
        postprocessed_xarray: The xarray Dataset with the data to plot.
        error_to_plot (str): Which error to plot. Default is 'mae'. Other options: 'bias', 'mae_score'
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
        marker_size: The size of the markers in the plot.
        title_fontsize: The fontsize of the title.


    Returns:
        The matplotlib PathCollection, pcs, as a dictionary with one key for each output variable name.
        Each value pcs[output_var_name] is a dict that has the following keys:
            - 'targets': the matplotlib PathCollection for the targets of the output variable
            - 'preds': the matplotlib PathCollection for the predictions of the output variable
            - error_to_plot: the matplotlib PathCollection for the error of the output variable
    """
    # Sample a random subset of the snapshots
    snaps = sorted(random.sample(range(postprocessed_xarray.dims[data_dim]), num_snapshots_to_plot))
    proj = ccrs.PlateCarree()
    label_fontsize = title_fontsize // 1.5
    ds_snaps = postprocessed_xarray.isel({'snapshot': snaps})
    kwargs = dict(
        x=longitude_dim,
        y=latitude_dim,
        row='plotting_dim',
        col=data_dim,
        transform=proj,
        subplot_kws={'projection': proj},
        s=marker_size,  # marker size
        robust=robust,
        cbar_kwargs={'shrink': 0.5,  # make cbar smaller/larger
                     'pad': 0.01,  # padding between right-ost subplot and cbar
                     'fraction': 0.05}
    )
    output_vars = postprocessed_xarray.variable_names.split(";")
    vmin = vmax = None
    ps = dict()
    for var in output_vars:
        d1, d2, d3 = getattr(ds_snaps, f'{var}_targets'), getattr(ds_snaps, f'{var}_preds'), getattr(ds_snaps,
                                                                                                     f'{var}_{error_to_plot}')
        plotting_ds = xr.concat([d1, d2, d3],
                                pd.Index(["Targets", "Preds", error_to_plot.upper()], name="plotting_dim"))
        p_target = xr.plot.scatter(plotting_ds, hue='plotting_dim', cmap=var_names_to_cmap[var], **kwargs)

        for ax in list(p_target.axes.flat):
            ax.coastlines()

        # Add title to variable subplots at the middle top
        title_v = f"{var_names_to_clean_name()[var]}"  # (${error_to_plot.upper()}={snapshot_err:.3f}$)"
        p_target.fig.suptitle(title_v, fontsize=title_fontsize, y=0.9)

    return ps


def data_snapshots_plotting(postprocessed_xarray: xr.Dataset,
                            error_to_plot: str = "mae",
                            vars_to_plot: List[str] = 'all',
                            num_snapshots_to_plot: int = 5,
                            snapshots_to_plot: List[int] = None,
                            data_dim: str = 'snapshot',
                            longitude_dim: str = 'longitude',
                            latitude_dim: str = 'latitude',
                            robust: bool = True,
                            same_colorbar_for_preds_and_targets: bool = True,
                            only_plot_preds: bool = False,
                            plot_error: bool = True,
                            marker_size: int = 2,
                            title: str = "",
                            title_fontsize: int = 18,
                            cmap='auto',
                            seed=7,
                            ):
    """

    Args:
        postprocessed_xarray: The xarray Dataset with the data to plot.
        error_to_plot (str): Which error to plot. Default is 'mae'. Other options: 'bias', 'mae_score'
        vars_to_plot: Which output variables to plot. Default is 'all' (plot all output variables).
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
        marker_size: The size of the markers in the plot.
        title_fontsize: The fontsize of the title.
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
    proj = ccrs.PlateCarree()
    label_fontsize = title_fontsize // 1.5
    ds_snaps = postprocessed_xarray.isel({'snapshot': snaps})
    kwargs = dict(
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
    if vars_to_plot == "all":
        output_vars = postprocessed_xarray.variable_names.split(";")
    else:
        assert isinstance(vars_to_plot, list), f"vars_to_plot must be a list, but is {type(vars_to_plot)}"
        output_vars = vars_to_plot
    p_target = p_err = vmin = vmax = None
    ps = dict()
    for var in output_vars:
        cmap = p_cmap = var_names_to_cmap[var] if cmap == 'auto' else cmap
        if not only_plot_preds:
            p_target = xr.plot.scatter(hue=f'{var}_targets', cmap=cmap, **kwargs)
            if same_colorbar_for_preds_and_targets:
                # Set the same colorbar for the predictions and targets subplots
                vmin, vmax = p_target.cbar.vmin, p_target.cbar.vmax
                p_cmap = p_target.cbar.cmap
            # Set coastlines of all axes
            for ax in list(p_target.axes.flat):
                ax.coastlines()

        p_pred = xr.plot.scatter(hue=f'{var}_preds', cmap=p_cmap, vmin=vmin, vmax=vmax, **kwargs)
        if plot_error and not only_plot_preds:
            p_err = xr.plot.scatter(hue=f'{var}_{error_to_plot}', **kwargs)
            for ax in list(p_err.axes.flat):
                ax.coastlines()
        ps[var] = {'targets': p_target, 'preds': p_pred, error_to_plot: p_err}

        # Set row names (ylabel) for the leftmost subplot of each row
        dtypes = ['preds'] if only_plot_preds else ['targets', 'preds', error_to_plot] if plot_error else ['targets', 'preds']
        for dtype in dtypes:
            pc = ps[var][dtype]
            ylabel = dtype.upper() if dtype == error_to_plot else dtype.capitalize()
            # pc.axes.flat[0].set_ylabel(ylabel, size=label_fontsize)
            # Edit the colorbar
            pc.cbar.set_label(ylabel, size=label_fontsize)
            pc.cbar.ax.tick_params(labelsize=label_fontsize // 1.7)

        # Set coastlines of all axes
        for ax in list(p_pred.axes.flat):
            ax.coastlines()

        if only_plot_preds:
            p_pred.fig.suptitle(title, fontsize=title_fontsize, y=0.9)
        else:
            # Add title to variable subplots at the middle top
            title_v = f"{var_names_to_clean_name()[var]} {title}"  # (${error_to_plot.upper()}={snapshot_err:.3f}$)"
            p_target.fig.suptitle(title_v, fontsize=title_fontsize, y=0.9)

        # remove snapshot = snapshot_id title from middle plots (already included in targets (top row))
        for ax in list(p_pred.axes.flat):
            ax.set_title('')
        if plot_error and not only_plot_preds:
            # Add bulk errors (per snapshot) as title to error subplots
            for i, (ax, snap_num) in enumerate(zip(p_err.axes.flat, p_err.col_names)):
                snapshot_mae = float(getattr(p_err.data, f'{var}_mae').sel({data_dim: snap_num}).mean().data)
                snapshot_bias = float(getattr(p_err.data, f'{var}_bias').sel({data_dim: snap_num}).mean().data)
                snapshot_bias_f = "" if var == 'pr' else f"Bias$={snapshot_bias:.3f}$"
                snapshot_mae_f = f"{snapshot_mae:.6f}" if var == 'pr' else f"{snapshot_mae:.3f}"
                ax.set_title(
                    f"$MAE={snapshot_mae_f}$, {snapshot_bias_f}",
                    fontsize=label_fontsize,
                )

    return ps


def adjust_subplots_spacing(
        left=0.125,  # the left side of the subplots of the figure
        right=0.9,  # the right side of the subplots of the figure
        bottom=0.1,  # the bottom of the subplots of the figure
        top=0.9,  # the top of the subplots of the figure
        wspace=0.2,  # the amount of width reserved for blank space between subplots
        hspace=0.2,  # the amount of height reserved for white space between subplots
):
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

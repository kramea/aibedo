import logging
from abc import ABC
from os.path import join
from typing import Optional, Union, Dict, Iterable, Sequence, List, Callable, Tuple
import numpy as np
import xarray as xr
import torch
from torch import Tensor
from aibedo.utilities.utils import get_logger, stem_var_id


def get_mean_and_std_xarray(data_dir: str, files_id: str) -> Tuple[xr.Dataset, xr.Dataset]:
    """ Get the climatology monthly mean and std of the CMIP6 ESMs used  """
    #          = f'ymonmean.1980_2010.{files_id}.CMIP6.historical.ensmean.Output.nc'
    files_id = files_id.strip('.')
    if files_id == '':
        # mean_fname = 'grid_1deg.ymonmean.1980_2010.CMIP6.historical.ensmean.Output.nc'
        mean_fname = 'grid_CESM_f09.ymonmean.1980_2010.CMIP6.historical.ensmean.Output.nc'
    else:
        mean_fname = f'ymonmean.1980_2010.{files_id}.CMIP6.historical.ensmean.Output.PrecipCon.nc'
        mean_fname = mean_fname.replace('..', '.')  # fix if .. is in the filename
    mean_ds = xr.open_dataset(join(data_dir, mean_fname))
    std_ds = xr.open_dataset(join(data_dir, mean_fname.replace('ymonmean', 'ymonstd')))
    return mean_ds, std_ds


def get_variable_stats(var_id: str, data_dir: str, files_id: str) -> (Tensor, Tensor):
    """
    Returns the monthly mean and std of the variable var_id

    Args:

        var_id (str): The variable id (e.g. 'tas', 'pr', 'ps', 'evspsbl')
    Returns:
        The monthly mean (climatology) and std of the variable var_id
    """
    mean_ds, std_ds = get_mean_and_std_xarray(data_dir, files_id)
    var_id = stem_var_id(var_id)  # e.g. 'pr' instead of 'pr_pre'
    monthly_means = torch.from_numpy(getattr(mean_ds, var_id).values)  # (12, #pixels)
    monthly_stds = torch.from_numpy(getattr(std_ds, var_id).values)  # e.g std_ds.pr.values
    # mean_dict = {m: torch.from_numpy(p) for m, p in zip(np.arange(12), monthly_means)}
    # std_dict = {m: torch.from_numpy(p) for m, p in zip(np.arange(12), monthly_stds)} # 0,1,2,3,4,5,6,7,8,9,10,11
    return monthly_means, monthly_stds


def get_clim_err(err_id: str, data_dir: str, files_id: str) -> Tensor:
    """ err_id should be one of 'PE', 'PS', 'Precip """
    err_id = err_id.replace('_clim_err', '')
    files_id = files_id.replace('isosph', 'isoph').strip('.')
    files_id = 'isoph6' if files_id in ['isoph', '', 'compress.isoph'] else files_id.replace('compress.', '')
    # files_id = 'compress.' + files_id
    # fix inconsistent naming issue of isoph instead of isosph
    err = np.load(join(data_dir, f'CMIP6_{err_id}_clim_err.{files_id}.npy'))  # (12,)
    return torch.from_numpy(err)


def standardize(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """
    Standardize the input x by subtracting the mean and dividing by the std.
    """
    return (x - mean) / std


def deanomalize(x: Tensor, mean: Tensor) -> Tensor:
    """
    De-anomalize the input x by adding the (monthly/climatology) mean.
    """
    return x + mean

def destandardize(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """
    Destandardize the input x
    """
    return x * std + mean


def rescale_nonorm_vars(nonorm_data: Dict[str, Tensor]):
    if 'pr_nonorm' in nonorm_data:
        nonorm_data['pr_nonorm'] = nonorm_data['pr_nonorm'] / 8.64e4  # convert from mm/day to kg/m2/s
    if 'ps_nonorm' in nonorm_data:
        nonorm_data['ps_nonorm'] = nonorm_data['ps_nonorm'] * 100  # convert from hPa to Pa
    if 'psl_nonorm' in nonorm_data:
        nonorm_data['psl_nonorm'] = nonorm_data['psl_nonorm'] * 100  # convert from hPa to Pa
    return nonorm_data

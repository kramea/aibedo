"""
Author: Salva Rühling Cachay
"""
import functools
import glob
import logging
import math
import os
import random

from types import SimpleNamespace
from typing import Union, Sequence, List, Dict, Optional, Callable, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, open_dict, OmegaConf
from torch import Tensor
from pytorch_lightning.utilities import rank_zero_only


def no_op(*args, **kwargs):
    pass


def identity(X):
    return X


def get_identity_callable(*args, **kwargs) -> Callable:
    return identity


def get_activation_function(name: str, functional: bool = False, num: int = 1):
    """ Returns the activation function with the given name. """
    name = name.lower().strip()

    def get_functional(s: str) -> Optional[Callable]:
        return {"softmax": F.softmax, "relu": F.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid,
                "identity": nn.Identity(),
                None: None, 'swish': F.silu, 'silu': F.silu, 'elu': F.elu, 'gelu': F.gelu, 'prelu': nn.PReLU(),
                }[s]

    def get_nn(s: str) -> Optional[Callable]:
        return {"softmax": nn.Softmax(dim=1), "relu": nn.ReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(),
                "identity": nn.Identity(), 'silu': nn.SiLU(), 'elu': nn.ELU(), 'prelu': nn.PReLU(),
                'swish': nn.SiLU(), 'gelu': nn.GELU(),
                }[s]

    if num == 1:
        return get_functional(name) if functional else get_nn(name)
    else:
        return [get_nn(name) for _ in range(num)]


def get_normalization_layer(name, dims, num_groups=None, *args, **kwargs):
    """ Returns the normalization layer with the given name.

    Args:
        name: name of the normalization layer. Must be one of ['batch_norm', 'layer_norm' 'group', 'instance', 'none']
    """
    if not isinstance(name, str) or name.lower() == 'none':
        return None
    elif 'batch_norm' == name:
        return nn.BatchNorm1d(num_features=dims, *args, **kwargs)
    elif 'layer_norm' == name:
        return nn.LayerNorm(dims, *args, **kwargs)
    elif 'rms_layer_norm' == name:
        from aibedo.utilities.normalization import RMSLayerNorm
        return RMSLayerNorm(dims, *args, **kwargs)
    elif 'instance' in name:
        return nn.InstanceNorm1d(num_features=dims, *args, **kwargs)
    elif 'group' in name:
        if num_groups is None:
            # find an appropriate divisor (not robust against weird dims!)
            pos_groups = [int(dims / N) for N in range(2, 17) if dims % N == 0]
            if len(pos_groups) == 0:
                raise NotImplementedError(f"Group norm could not infer the number of groups for dim={dims}")
            num_groups = max(pos_groups)
        return nn.GroupNorm(num_groups=num_groups, num_channels=dims)
    else:
        raise ValueError("Unknown normalization name", name)


def get_loss(name, reduction='mean'):
    """ Returns the loss function with the given name. """
    name = name.lower().strip().replace('-', '_')
    if name in ['l1', 'mae', "mean_absolute_error"]:
        loss = nn.L1Loss(reduction=reduction)
    elif name in ['l2', 'mse', "mean_squared_error"]:
        loss = nn.MSELoss(reduction=reduction)
    elif name in ['smoothl1', 'smooth']:
        loss = nn.SmoothL1Loss(reduction=reduction)
    else:
        raise ValueError(f'Unknown loss function {name}')
    return loss


def to_dict(obj: Optional[Union[dict, SimpleNamespace]]):
    if obj is None:
        return dict()
    elif isinstance(obj, dict):
        return obj
    else:
        return vars(obj)


def to_DictConfig(obj: Optional[Union[List, Dict]]):
    """ Tries to convert the given object to a DictConfig. """
    if isinstance(obj, DictConfig):
        return obj

    if isinstance(obj, list):
        try:
            dict_config = OmegaConf.from_dotlist(obj)
        except ValueError as e:
            dict_config = OmegaConf.create(obj)

    elif isinstance(obj, dict):
        dict_config = OmegaConf.create(obj)

    else:
        dict_config = OmegaConf.create()  # empty

    return dict_config


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


#####
# The following two functions extend setattr and getattr to support chained objects, e.g. rsetattr(cfg, optim.lr, 1e-4)
# From https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def rhasattr(obj, attr, *args):
    def _hasattr(obj, attr):
        return hasattr(obj, attr, *args)

    return functools.reduce(_hasattr, [obj] + attr.split('.'))


# Errors
def raise_error_if_invalid_value(value: Any, possible_values: Sequence[Any], name: str = None):
    """ Raises an error if the given value (optionally named by `name`) is not one of the possible values. """
    if value not in possible_values:
        name = name or (value.__name__ if hasattr(value, '__name__') else 'value')
        raise ValueError(f"{name} must be one of {possible_values}, but was {value}")
    return value


# Random seed (if not using pytorch-lightning)
def set_seed(seed, device='cuda'):
    """
    Sets the random seed for the given device.
    If using pytorch-lightning, preferably to use pl.seed_everything(seed) instead.
    """
    # setting seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device != 'cpu':
        torch.cuda.manual_seed(seed)


# Checkpointing
def get_epoch_ckpt_or_last(ckpt_files: List[str], epoch: int = None):
    if epoch is None:
        if 'last.ckpt' in ckpt_files:
            model_ckpt_filename = 'last.ckpt'
        else:
            ckpt_epochs = [int(name.replace('epoch', '')[:3]) for name in ckpt_files]
            # Use checkpoint with latest epoch if epoch is not specified
            max_epoch = max(ckpt_epochs)
            model_ckpt_filename = [name for name in ckpt_files if str(max_epoch) in name][0]
        logging.warning(f"Multiple ckpt files exist: {ckpt_files}. Using latest epoch: {model_ckpt_filename}")
    else:
        # Use checkpoint with specified epoch
        model_ckpt_filename = [name for name in ckpt_files if str(epoch) in name]
        if len(model_ckpt_filename) == 0:
            raise ValueError(f"There is no ckpt file for epoch={epoch}. Try one of the ones in {ckpt_files}!")
        model_ckpt_filename = model_ckpt_filename[0]
    return model_ckpt_filename


def get_local_ckpt_path(config: DictConfig, **kwargs):
    ckpt_direc = config.callbacks.model_checkpoint.dirpath
    if not os.path.isdir(ckpt_direc):
        logging.warning(f"Ckpt directory {ckpt_direc} does not exist. Are you sure the ckpt is on this file-system?.")
        return None
    ckpt_filenames = [f for f in os.listdir(ckpt_direc) if os.path.isfile(os.path.join(ckpt_direc, f))]
    filename = get_epoch_ckpt_or_last(ckpt_filenames, **kwargs)
    return os.path.join(ckpt_direc, filename)


def get_any_ensemble_id(data_dir, ESM_NAME: str) -> str:
    sphere = "isosph"  # change here to isosph5 for level 5 runs
    prefix = f"compress.{sphere}.{ESM_NAME}.historical"
    if os.path.isfile(os.path.join(data_dir, f"{prefix}.r1i1p1f1.Input.Exp8_fixed.nc")):
        fname = f"{prefix}.r1i1p1f1.Input.Exp8_fixed.nc"
    else:
        curdir = os.getcwd()
        os.chdir(data_dir)
        files = glob.glob(f"{prefix}.*.Input.Exp8_fixed.nc")
        fname = files[0]
        os.chdir(curdir)
    return fname

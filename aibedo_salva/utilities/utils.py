"""
Author: Salva Rühling Cachay
"""
import functools
import logging
import math
import os

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
    if not isinstance(name, str) or name.lower() == 'none':
        return None
    elif 'batch_norm' == name:
        return nn.BatchNorm1d(num_features=dims, *args, **kwargs)
    elif 'layer_norm' == name:
        return nn.LayerNorm(dims, *args, **kwargs)
    elif 'inst' in name:
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


# Errors
def raise_error_if_invalid_value(value: Any, possible_values: Sequence[Any], name: str = None):
    if value not in possible_values:
        name = name or (value.__name__ if hasattr(value, '__name__') else 'value')
        raise ValueError(f"{name} must be one of {possible_values}, but was {value}")

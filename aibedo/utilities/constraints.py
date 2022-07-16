import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def nonnegative_precipitation(precipitation: Tensor) -> Tensor:
    """
    Constraint: precipitation must be nonnegative.

    Args:
        precipitation (Tensor): precipitation tensor (in raw/denormalized scale!)
    """
    return F.relu(precipitation)


def global_moisture_constraint(precipitation: Tensor, evaporation: Tensor, PE_err: Tensor) -> float:
    """

    Args:
        precipitation (Tensor): precipitation tensor (in raw/denormalized scale!)
        evaporation (Tensor): evaporation tensor (in raw/denormalized scale!)
        PE_err:

    Returns:
        The scalar constraint error (mean over batch)

    """
    # Same as (can be quickly checked with assert torch.isclose(loss_pr, err), for example):
    #  loss_pr, batch_size = 0.0, precipitation.shape[0]
    #  for i in range(batch_size):
    #       loss_pr += (precipitation[i, :] - evaporation[i, :]).mean() - PE_err[i].mean()
    #  loss_pr =/ batch_size
    err = ((precipitation - evaporation).mean(dim=0) - PE_err.mean(dim=0)).mean()
    return err


def global_moisture_constraint_soft_loss(*args, **kwargs) -> float:
    """ Soft loss to be used for backprop (for global moisture constraint) """
    return torch.abs(global_moisture_constraint(*args, **kwargs))


def mass_conservation_constraint(surface_pressure: Tensor, PS_err: Tensor) -> float:
    """

    Args:
        surface_pressure (Tensor): pressure tensor (in raw/denormalized scale!)
        PS_err:

    Returns:
        The scalar constraint error (mean over batch)

    """
    # Same as (can be quickly checked with assert torch.isclose(loss_ps, err), for example):
    #  loss_ps, batch_size = 0.0, surface_pressure.shape[0]
    #  for i in range(batch_size):
    #       loss_ps += surface_pressure[i, :].mean() - PS_err[i].mean()
    #  loss_ps =/ batch_size
    err = (surface_pressure.mean(dim=0) - PS_err.mean(dim=0)).mean()
    return err


def mass_conservation_constraint_soft_loss(*args, **kwargs) -> float:
    """ Soft loss to be used for backprop (for mass conservation constraint) """
    return torch.abs(mass_conservation_constraint(*args, **kwargs))

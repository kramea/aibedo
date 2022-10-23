import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

AUXILIARY_VARS = [
    'evspsbl_nonorm',  # evaporation
    'hfss_nonorm',  # heat flux sea surface
    'netTOARad_nonorm',  # top-of-atmosphere level radiation (shortwave)
    'netSurfRad_nonorm'  # surface level radiation (longwave)
]


def nonnegative_precipitation(precipitation: Tensor) -> Tensor:
    """
    Constraint: precipitation must be nonnegative.

    Args:
        precipitation (Tensor): precipitation tensor (in raw/denormalized scale!)
    """
    return F.relu(precipitation)


def precipitation_energy_budget_constraint(
        precipitation: Tensor,
        sea_surface_heat_flux: Tensor,
        toa_sw_net_radiation: Tensor,
        surface_lw_net_radiation: Tensor,
        PR_Err: Tensor
) -> Tensor:
    """

    Args:
        precipitation:
        sea_surface_heat_flux:
        toa_sw_net_radiation:
        surface_lw_net_radiation:
        PR_Err:

    Returns:

    """
    # loss_peb, batch_size = 0.0, precipitation.shape[0]
    # for i in range(batch_size):
    #    pr = precipitation[i, :] * 2.4536106 * 1_000_000.0
    #    actual = pr + sea_surface_heat_flux[i, :] + toa_sw_net_radiation[i, :] - surface_lw_net_radiation[i, :]
    #    loss_peb += actual.mean() - PR_Err[i].mean()
    # loss_peb /= batch_size
    spatial_dims = (-2, -1) if precipitation.dim() == 3 else -1  # for 2D data dim = 3, for spherical data dim=2
    pr_scaled = precipitation * 2.4536106 * 1_000_000
    actual = (
            pr_scaled + sea_surface_heat_flux + toa_sw_net_radiation - surface_lw_net_radiation
    ).mean(dim=spatial_dims)
    loss_peb2 = actual - PR_Err
    return loss_peb2  # .mean()  # mean over batch dimension


def global_moisture_constraint(precipitation: Tensor, evaporation: Tensor, PE_err: Tensor) -> Tensor:
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
    spatial_dims = (-2, -1) if precipitation.dim() == 3 else -1  # for 2D data dim = 3, for spherical data dim=2
    err = (precipitation - evaporation).mean(dim=spatial_dims) - PE_err  # mean is over spatial dimension (1 or -1)
    return err  # .mean()  # return average constraint error over batch


def global_moisture_constraint_soft_loss(*args, **kwargs) -> float:
    """ Soft loss to be used for backprop (for global moisture constraint) """
    return torch.abs(global_moisture_constraint(*args, **kwargs))


def mass_conservation_constraint(surface_pressure: Tensor, PS_err: Tensor) -> Tensor:
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
    spatial_dims = (-2, -1) if surface_pressure.dim() == 3 else -1  # for 2D data dim = 3, for spherical data dim=2
    err = surface_pressure.mean(dim=spatial_dims) - PS_err  # take pressure mean over the spatial dimension
    return err  # .mean()


def mass_conservation_constraint_soft_loss(*args, **kwargs) -> float:
    """ Soft loss to be used for backprop (for mass conservation constraint) """
    return torch.abs(mass_conservation_constraint(*args, **kwargs))

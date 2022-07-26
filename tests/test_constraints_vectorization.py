import torch

from aibedo.utilities.constraints import precipitation_energy_budget_constraint, global_moisture_constraint, \
    mass_conservation_constraint


def test_precipitation_energy_budget_constraint(n_trials=10):
    for i in range(n_trials):
        ngrids = i * 1000 + 100
        batch_size = i * 10 + 5
        precipitation = torch.randn(batch_size, ngrids)
        precipitation += torch.min(precipitation)

        sea_surface_heat_flux = torch.randn(batch_size, ngrids)
        toa_sw_net_radiation = torch.randn(batch_size, ngrids)
        surface_lw_net_radiation = torch.randn(batch_size, ngrids)
        PR_Err = torch.randn(batch_size)

        loss_peb = 0.0
        for i in range(batch_size):
            pr = precipitation[i, :] * 2.4536106 * 1_000_000.0
            actual = pr + sea_surface_heat_flux[i, :] + toa_sw_net_radiation[i, :] - surface_lw_net_radiation[i, :]
            loss_peb += actual.mean() - PR_Err[i].mean()
        loss_peb /= batch_size

        # Vectorized version:
        loss_peb2 = precipitation_energy_budget_constraint(
            precipitation, sea_surface_heat_flux, toa_sw_net_radiation, surface_lw_net_radiation, PR_Err
        )
        assert torch.isclose(loss_peb2, loss_peb), f"Losses are not the same!: {loss_peb} vs {loss_peb2}"


def test_global_moisture_constraint(n_trials=10):
    for i in range(n_trials):
        ngrids = i * 1000 + 100
        batch_size = i * 10 + 5
        precipitation = torch.randn(batch_size, ngrids)
        precipitation += torch.min(precipitation)  # non-negative

        evaporation = torch.randn(batch_size, ngrids)
        PE_err = torch.randn(batch_size)

        loss = 0.0
        for i in range(batch_size):
            loss += (precipitation[i, :] - evaporation[i, :]).mean() - PE_err[i].mean()
        loss /= batch_size

        # Vectorized version:
        loss_vec = global_moisture_constraint(
            precipitation, evaporation=evaporation, PE_err=PE_err
        )
        assert torch.isclose(loss, loss_vec, rtol=1e-5), f"Losses are not the same!: {loss} vs {loss_vec}"


def test_mass_conservation_constraint(n_trials=10):
    for i in range(n_trials):
        ngrids = i * 1000 + 100
        batch_size = i * 10 + 5
        surface_pressure = torch.randn(batch_size, ngrids) * 10_000
        surface_pressure += torch.min(surface_pressure)  # non-negative

        PS_err = torch.randn(batch_size)

        loss = 0.0
        for i in range(batch_size):
            loss += surface_pressure[i, :].mean() - PS_err[i].mean()
        loss /= batch_size

        # Vectorized version:
        loss_vec = mass_conservation_constraint(
            surface_pressure, PS_err=PS_err
        )
        assert torch.isclose(loss, loss_vec), f"Losses are not the same!: {loss} vs {loss_vec}"


if __name__ == '__main__':
    test_precipitation_energy_budget_constraint()
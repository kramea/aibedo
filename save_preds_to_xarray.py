import xarray as xr
from aibedo.utilities.wandb_api import reload_checkpoint_from_wandb

cur_run_id = None


def get_model_and_dm_from_run_id(run_id, overrides):
    global cur_run_id
    cur_run_id = run_id
    values = reload_checkpoint_from_wandb(run_id=run_id, project='AIBEDO', override_key_value=overrides,
                                          try_local_recovery=False)
    return values['model'], values['datamodule'], values['config']


def get_predictions_xarray(
        run_id, overrides, split='predict', also_errors=False, variables='all', **kwargs
) -> xr.Dataset:
    model, dm, cfg = get_model_and_dm_from_run_id(run_id, overrides)
    print('physics_loss_weights:', cfg.model.physics_loss_weights)
    dm.setup(stage=split)
    dataloader = dm.predict_dataloader()
    predictions_xarray = dm.get_predictions_xarray(model, dataloader=dataloader,
                                                   also_errors=also_errors,
                                                   variables=variables,
                                                   **kwargs)
    del model, dm, cfg
    return predictions_xarray


if __name__ == "__main__":
    DATA_DIR = "/network/scratch/s/salva.ruhling-cachay/TMP_climart/aibedo/data"
    overrides = [f'datamodule.num_workers={6}',
                 'datamodule.eval_batch_size=4',
                 'verbose=False',
                 '++model.use_auxiliary_vars=False',
                 f'datamodule.data_dir={DATA_DIR}',
                 ]
    run_ids = ["37tl5tsy", '34q3echu']
    run_ids = ["ccmopc1x", '1ul6rx81']
    run_ids = [
        ('ubjfpr0f', '_MLP_spherical_4constraints'),  # mlp spherical
        ('3b6fg6sb', '_MLP_spherical_0constraints'),  # mlp spherical
        ('3lax8md0', '_FNO_spherical'),  # fno spherical, seed 7
    ]
    predict_overrides_sets = [['++datamodule.prediction_data=val'],
                              ['++datamodule.prediction_data=same_as_test', 'datamodule.partition=[0.85, 0.15, era5]']]
    for run_id in run_ids:
        run_id_save_kwargs = ''
        if isinstance(run_id, tuple):
            run_id, run_id_save_kwargs = run_id
        for predict_set_overrides in predict_overrides_sets:
            predict_set_overrides += overrides
            pds = get_predictions_xarray(run_id,
                                         predict_set_overrides,
                                         also_errors=True,
                                         also_targets=True,
                                         split='predict',
                                         return_normalized_outputs=True, variables='all')
            dataset_name = pds.attrs['dataset_name']
            fname = f"predictions_{dataset_name}_{run_id}{run_id_save_kwargs}.nc"
            fpath = f"{DATA_DIR.replace('data', 'out_dir/preds')}/{fname}"
            pds.to_netcdf(fpath)
            # print in a box of text with stars on first line and dashes on second line
            print(f"{'*' * len(fpath)}")
            print(f"Saved predictions to:\n{fpath}")
            print(f"{'-' * len(fpath)}")

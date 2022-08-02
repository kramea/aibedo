import glob
import os
import sys
from os.path import join

import hydra
from hydra.core.global_hydra import GlobalHydra

import aibedo.constants
from aibedo.train import run_model
from aibedo.utilities.utils import get_any_ensemble_id


def single_esm_training_run(overrides, ESM: str):
    hydra.initialize(config_path="aibedo/configs", version_base=None)
    try:
        config = hydra.compose(config_name="main_config.yaml", overrides=overrides)
    finally:
        GlobalHydra.instance().clear()  # always clean up global hydra
    config.datamodule.input_filename = get_any_ensemble_id(config.datamodule.data_dir, ESM)
    return run_model(config)


def all_esm_runs():
    ESMs = aibedo.constants.CLIMATE_MODELS_ALL
    i = 1
    if len(sys.argv) <= i:
        test_set = 'era5'
    elif 'era' in sys.argv[i]:
        test_set, i = 'era5', i + 1
    elif 'merra' in sys.argv[i]:
        test_set, i = 'merra2', i + 1
    else:
        test_set = 'era5'

    if len(sys.argv) <= i:
        pass
    elif 'first' in sys.argv[i]:
        ESMs = ESMs[:int(sys.argv[i][5:])]
        args = sys.argv[i + 1:]
    elif 'last' in sys.argv[i]:
        ESMs = ESMs[-int(sys.argv[i][4:]):]
        args = sys.argv[i + 1:]
    else:
        args = sys.argv[i:]

    main_overrides = [
                         f'datamodule.partition=[0.85, 0.15, {test_set}]',
                     ] + list(args)
    print(main_overrides)

    for ESM_NAME in ESMs:
        single_esm_training_run(overrides=main_overrides, ESM=ESM_NAME)


if __name__ == "__main__":
    all_esm_runs()

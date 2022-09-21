import glob
import os
import sys
from os.path import join

import hydra
from hydra.core.global_hydra import GlobalHydra

import aibedo.constants
from aibedo.train import run_model


def single_esm_training_run(overrides, ESM: str):
    for arg in overrides:
        assert 'datamodule.esm_for_training=' not in arg, f"ESM should not be specified in overrides when calling this script: {arg}"

    esm_overrides = overrides + [f'datamodule.esm_for_training={ESM}']

    hydra.initialize(config_path="aibedo/configs", version_base=None)
    try:
        config = hydra.compose(config_name="main_config.yaml", overrides=esm_overrides)
    finally:
        GlobalHydra.instance().clear()  # always clean up global hydra
    # data_file_id = get_files_prefix(config.datamodule)
    # config.datamodule.input_filename = get_any_ensemble_id(config.datamodule.data_dir, ESM, files_id=data_file_id)
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

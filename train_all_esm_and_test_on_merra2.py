import glob
import os
import sys
from os.path import join

import hydra
from hydra.core.global_hydra import GlobalHydra

import aibedo.constants
from aibedo.train import run_model


def single_esm_training_run(overrides, ESM: str):
    hydra.initialize(config_path="aibedo/configs", version_base=None)
    try:
        config = hydra.compose(config_name="main_config.yaml", overrides=overrides)
    finally:
        GlobalHydra.instance().clear()  # always clean up global hydra
    config.datamodule.input_filename = get_any_ensemble_id(config.datamodule.data_dir, ESM)
    return run_model(config)


def get_any_ensemble_id(data_dir, ESM_NAME: str) -> str:
    sphere = "isosph"   # change here to isosph5 for level 5 runs
    prefix = f"compress.{sphere}.{ESM_NAME}.historical"
    if os.path.isfile(join(data_dir, f"{prefix}.r1i1p1f1.Input.Exp8_fixed.nc")):
        fname = f"{prefix}.r1i1p1f1.Input.Exp8_fixed.nc"
    else:
        curdir = os.getcwd()
        os.chdir(data_dir)
        files = glob.glob(f"{prefix}.*.Input.Exp8_fixed.nc")
        fname = files[0]
        os.chdir(curdir)
    return fname


def all_esm_runs():
    ESMs = aibedo.constants.CLIMATE_MODELS_ALL
    if len(sys.argv) <= 1:
        pass
    elif 'first' in sys.argv[1]:
        ESMs = ESMs[:int(sys.argv[1][5:])]
        args = sys.argv[2:]
    elif 'last' in sys.argv[1]:
        ESMs = ESMs[-int(sys.argv[1][4:]):]
        args = sys.argv[2:]
    else:
        args = sys.argv[1:]

    main_overrides = [
        'datamodule.partition=[0.85, 0.15, merra2]',
    ] + list(args)
    print(main_overrides)

    for ESM_NAME in ESMs:
        single_esm_training_run(overrides=main_overrides, ESM=ESM_NAME)


if __name__ == "__main__":
    all_esm_runs()

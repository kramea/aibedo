import logging
import os
import sys
import wandb

from aibedo.constants import CLIMATE_MODELS_ALL
from aibedo.test import reload_and_test_model

os.environ['WANDB_SILENT'] = "true"

if __name__ == '__main__':
    api = wandb.Api(timeout=60)

    entity, project = 'salv47', "AIBEDO"
    prefix = entity + '/' + project
    OVERRIDE = True

    runs = api.runs(prefix, order='-created_at')
    TAG = "esm-train-vs-ra"
    for i, run in enumerate(runs):
        if run.state in ['running', 'crashed']:  # "finished":
            continue
        if TAG not in run.tags:
            continue
        for predict_set in CLIMATE_MODELS_ALL[:10]:  # ['same_as_test'] +
            if f'predict/{predict_set.upper()}/tas/rmse' in run.summary:
                continue

            override = [
                f'datamodule.partition=[0.85, 0.15,era5]',
                f'++datamodule.prediction_data={predict_set}',
                'trainer.gpus=-1']  # , 'callbacks=default.yaml']
            try:
                reload_and_test_model(run.id, override_kwargs=override,
                                      train_model=False, test_model=False, predict_model=True)
            except (ValueError, RuntimeError) as e:
                logging.warning(f"{e}")
                continue

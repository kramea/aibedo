import logging
import os
import sys
import wandb

from aibedo.test import reload_and_test_model

os.environ['WANDB_SILENT'] = "true"

if __name__ == '__main__':
    api = wandb.Api(timeout=40)
    if sys.argv[1]:
        test_set = sys.argv[1]
    else:
        test_set = 'merra2'

    entity, project = 'salv47', "AIBEDO"
    prefix = entity + '/' + project
    d_dir = None  # r"C:\Users\salva\PycharmProjects\Data\RT_ML"
    OVERRIDE = True

    runs = api.runs(prefix, order='-created_at')
    for i, run in enumerate(runs):
        if run.state in ['running', 'crashed']:  # "finished":
            continue
        if 'esm-train-vs-ra' not in run.tags:  # or i > 3:
            continue
        if f'test/{test_set.upper()}/mse_epoch' in run.summary:
            continue
        override = [f'datamodule.partition=[0.85, 0.15, {test_set}]',
                    "++model.use_auxiliary_vars=False",
                    'trainer.gpus=-1']  # , 'callbacks=default.yaml']
        try:
            reload_and_test_model(run.id, override_kwargs=override, test_model=True)
        except ValueError as e:
            logging.warning(f"{e}")
            continue

import os

import numpy as np
import wandb

api = wandb.Api(timeout=100)

runs = api.runs(f"salv47/AIBEDO", per_page=100)

for run in runs:
    if 'datamodule/input_filename' not in run.config or 'datamodule/esm_for_training' in run.config:
        continue
    ifn = run.config['datamodule/input_filename']
    if 'isosph' in ifn:
        esm = ifn.split('.')[2]
    else:
        esm = ifn.split('.')[0]
    print(esm)
    run.config["datamodule/esm_for_training"] = esm
    run.update()

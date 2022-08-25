import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from aibedo.constants import CLIMATE_MODELS_ALL
from aibedo.utilities.wandb_api import get_runs_df, hasnt_tags, groupby, has_tags
from aibedo.utilities.plotting import set_labels_and_ticks, plot_training_set_vs_test_performance

thresh = None
save_dir = None  # r"C:\Users\salva\OneDrive\Documentos\Bachelor\MILA\figures"
ORDER_BY_PERFORMANCE = True
metrics = ['val/mse_epoch', 'test/MERRA2/mse_epoch', 'test/ERA5/mse_epoch']
metrics = ['test/ERA5/mse_epoch']
metrics += ['test/ERA5/pr/rmse_epoch', 'test/ERA5/ps/rmse_epoch', 'test/ERA5/tas/rmse_epoch']

metrics_to_plot = []
vars = ['ps', 'pr', 'tas']
metric = 'rmse'
for var in ['ps', 'pr', 'tas']:
    metrics_to_plot += [f'predict/{esm}/{var}/{metric}' for esm in CLIMATE_MODELS_ALL if esm not in ['CanESM5', 'FGOALS-g3']]
filter_by_hparams = {
  #  'model/name': 'MLP',
}
tags = ["MLP-all-4-constraints", "MLP-baseline"]
filters = [
    'has_finished',
    has_tags(['esm-train-vs-ra']),
]

runs_df: pd.DataFrame = get_runs_df(
    hyperparam_filter=filter_by_hparams,
    get_metrics=True,
    run_pre_filters=filters,
    make_hashable_df=True,
    verbose=0)
runs_df = runs_df.dropna(subset=metrics)
keep_columns = ['model/name', 'id', 'datamodule/input_filename']
for tag in tags:
    runs_df_tag = runs_df[runs_df['tags'].str.contains(tag)]
    num_esms = runs_df['datamodule/input_filename'].nunique()
    ESMs = [esm.split('.')[2] for esm in runs_df['datamodule/input_filename'].unique()]
    grouped_runs_stats = groupby(runs_df_tag,
                                 group_by=['datamodule/input_filename'],
                                 keep_columns=list(keep_columns),
                                 metrics=list(metrics_to_plot))


    for i, var in enumerate(vars):
        fig, ax = plt.subplots(1, 1)
        err_matrix = np.ones([num_esms, num_esms])
        esm_to_idx = {esm: i for i, esm in enumerate(ESMs)}

        xlabels, errors, stds = [], [], []
        for j, run_group in grouped_runs_stats.iterrows():
            run_ESM = run_group['datamodule/input_filename'].split('.')[2]
            idx = esm_to_idx[run_ESM]
            for esm_other in ESMs:
                idx_other = esm_to_idx[esm_other]
                if f"predict/{esm_other}/{var}/{metric}/mean" in run_group.keys():
                    y = run_group[f"predict/{esm_other}/{var}/{metric}/mean"]
                    y_std = run_group[f"predict/{esm_other}/{var}/{metric}/std"]
                else:
                    continue
                if y != y: # or y_std != y_std:  # NaN
                    print('Oops',run_ESM, esm_other, y)
                    continue
                if thresh and y > thresh:
                    print(f"run {run_group} has bad {var} score={y}, skipping it")
                    continue
                print(0, run_ESM, 11, esm_other, 2222, y)
                err_matrix[idx, idx_other] = y
                #print(f"run {run_group} has {var} {esm_other} score={y}")
        ax.imshow(err_matrix)


    plt.show()

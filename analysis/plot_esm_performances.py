import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from aibedo.utilities.wandb_api import get_runs_df, hasnt_tags, groupby, has_tags
from aibedo.utilities.plotting import set_labels_and_ticks, plot_training_set_vs_test_performance

thresh = None
save_dir = None  # r"C:\Users\salva\OneDrive\Documentos\Bachelor\MILA\figures"
ORDER_BY_PERFORMANCE = True
metrics = ['val/mse_epoch', 'test/MERRA2/mse_epoch', 'test/ERA5/mse_epoch']
metrics = ['test/ERA5/mse_epoch']
metrics += ['test/ERA5/pr/rmse_epoch', 'test/ERA5/ps/rmse_epoch', 'test/ERA5/tas/rmse_epoch']

filter_by_hparams = {
  #  'model/name': 'MLP',
}
tags = ["MLP-all-4-constraints", "MLP-baseline"]
filters = [
    'has_finished',
    has_tags(['esm-train-vs-ra']),
    # hasnt_tags(['physics-constraints'])
]

runs_df: pd.DataFrame = get_runs_df(
    hyperparam_filter=filter_by_hparams,
    get_metrics=True,
    run_pre_filters=filters,
    make_hashable_df=True,
    verbose=0)
runs_df = runs_df.dropna(subset=metrics)
keep_cols = ['model/name', 'id', 'datamodule/input_filename']
for tag in tags:
    runs_df_tag = runs_df[runs_df['tags'].str.contains(tag)]
    print(runs_df_tag['tags'].unique(), runs_df_tag['tags'].nunique())
    plot_training_set_vs_test_performance(runs_df_tag,
                                          metrics_to_plot=metrics,
                                          keep_columns=keep_cols,
                                          title=tag,
                                          save_to_dir=save_dir,
                                          max_error_to_plot=thresh)
plt.show()

import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from aibedo.utilities.wandb_api import get_runs_df, hasnt_tags
from aibedo.utilities.plotting import plot_training_set_vs_test_performance

thresh = None
save_dir = None  # r"C:\Users\salva\OneDrive\Documentos\Bachelor\MILA\figures"
ORDER_BY_PERFORMANCE = True
metrics = ['val/mse_epoch', 'test/MERRA2/mse_epoch', 'test/ERA5/mse_epoch']

filter_by_hparams = {
    'datamodule/order': 6
}
filters = [
    'has_final_metric', hasnt_tags(['physics-constraints'])
]

runs_df: pd.DataFrame = get_runs_df(
    hyperparam_filter=filter_by_hparams,
    get_metrics=True,
    run_pre_filters=filters,
    make_hashable_df=True,
    verbose=0)
print([x[:30] for x in runs_df['model/name'].unique()])
group_key = 'group/v0'
keep_cols = ['model/name', 'id', 'datamodule/input_filename', group_key]
runs_df = runs_df.replace('NaN', float('NaN')).dropna(subset=metrics)[keep_cols + metrics]
groups = [g for g in runs_df[group_key].unique() if g == g]
print(f'There are {len(groups)} groups: {groups}')
for group in groups:
    group_runs = runs_df[getattr(runs_df, group_key) == group]
    if group_runs['datamodule/input_filename'].nunique() <= 5:
        print(f'Skipping group {group} because it has {group_runs["datamodule/input_filename"].nunique()}')
        continue
    plot_training_set_vs_test_performance(group_runs,
                                          metrics_to_plot=metrics,
                                          keep_columns=keep_cols,
                                          title=group,
                                          ylim=[0.3, None],
                                          save_to_dir=save_dir,
                                          max_error_to_plot=thresh)
plt.show()

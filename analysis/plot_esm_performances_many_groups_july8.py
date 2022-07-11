import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from aibedo_salva.utilities.wandb_api import get_runs_df, hasnt_tags, groupby
from aibedo_salva.utilities.plotting import set_labels_and_ticks, RollingCmaps, RollingLineFormats

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
    if group_runs['datamodule/input_filename'].nunique() <= 3:
        print(f'Skipping group {group} because it has {group_runs["datamodule/input_filename"].nunique()}')
        continue
    fig, axs = plt.subplots(len(metrics), 1)
    grouped_runs_stats = groupby(group_runs,
                                 group_by=['datamodule/input_filename'],
                                 keep_columns=keep_cols,
                                 metrics=metrics)

    for i, metric in enumerate(metrics):
        print(group, metric)
        ax = axs[i]
        xlabels, errors, stds = [], [], []
        for j, run_group in grouped_runs_stats.iterrows():
            run_ESM = run_group['datamodule/input_filename'].split('.')[2]
            y = run_group[f"{metric}/mean"]
            y_std = run_group[f"{metric}/std"]
            first_run_id = run_group['id']
            # print(run_ESM, y, y_std)
            if y != y or y_std != y_std:  # NaN
                continue

            if thresh and y > thresh:
                print(f"run {run_group} has bad score={y}, skipping it")
                continue
            errors.append(y)
            stds.append(y_std)
            xlabels += [run_ESM]

        colors_ESM = cm.rainbow(np.linspace(0, 1, len(xlabels)))
        if ORDER_BY_PERFORMANCE:  # order positions by performance (lowest to highest)
            errors, stds, xlabels, colors = map(list, (zip(*sorted(zip(errors, stds, xlabels, colors_ESM), key=lambda x: x[0]))))

        x_pos = np.arange((len(xlabels)))
        ax.bar(x_pos, errors, yerr=stds, color=colors, align='center', alpha=0.5, ecolor='black', capsize=10)

        metric_name = metric.upper().replace('_EPOCH', '').replace('TEST/', '').replace('/', ' ').replace('VAL', 'Val')
        save_to = f"{metric_name}.png"
        set_labels_and_ticks(
            ax,
            xlabel='', ylabel=metric_name,
            xticks=x_pos, xtick_labels=xlabels,
            xlabel_fontsize=14, ylabel_fontsize=14,
            xticks_fontsize=8, yticks_fontsize=14, x_ticks_rotation=10,
            ylim=[0.3, None],
            title=f"{group}",
            full_screen=False,
            show=False, legend=False, legend_loc='best',
            grid=True,
            save_to=os.path.join(save_dir, save_to) if save_dir else None
        )
plt.show()

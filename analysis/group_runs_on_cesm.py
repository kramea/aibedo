import logging

import pandas as pd
import wandb
import os
from aibedo_salva.utilities.wandb_api import get_runs_df, hasnt_tags, filter_wandb_runs

os.environ['WANDB_CONSOLE'] = 'off'

api = wandb.Api(timeout=100)

entity, project = 'salv47', "AIBEDO"
prefix = entity + '/' + project
version = 0

filter_by_hparams = {
    'datamodule/input_filename': "compress.isosph.CESM2.historical.r1i1p1f1.Input.Exp8_fixed.nc",
}
filters = [
    'has_finished', hasnt_tags(['physics-constraints'])
]

runs_df: pd.DataFrame = get_runs_df(
    hyperparam_filter=filter_by_hparams,
    get_metrics=False,
    run_pre_filters=filters,
    make_hashable_df=True,
    order='+summary_metrics.val/mse_epoch',
    verbose=0)

dont_care_vals = ['seed', 'name', 'tags', 'group',
                  'model/params_trainable', 'model/params_not_trainable', 'model/params_total',
                  'model/input_transform/output_normalization',
                  'model/monitor', 'model/_target_',
                  'datamodule/time_length',
                  'normalizer/data_dir', 'normalizer/verbose', "normalizer/_target_",
                  'datamodule/_target_', 'datamodule/data_dir',
                  'datamodule/eval_batch_size', 'datamodule/partition',  # dont care which test set / eval batch-size
                  'datamodule/target_variable', 'datamodule/verbose',
                  'datamodule/eval_batch_size', 'datamodule/validation_years',
                  'trainer/min_epochs',
                  'trainer/_target_', 'trainer/num_sanity_val_steps', 'trainer/gpus',
                  'trainer/resume_from_checkpoint',
                  'model_checkpoint/save_last', 'model_checkpoint/filename',
                  'model_checkpoint/auto_insert_metric_name',
                  'model_checkpoint/_target_', 'model_checkpoint/verbose', 'model_checkpoint/dirpath',
                  'model_checkpoint/mode']
NAN_DUMMY_VALUE = "NULL"  # should not be in dataframe/hyperparams
dont_care_vals = [x for x in dont_care_vals if x in runs_df.columns]
runs_df = runs_df.drop(columns=dont_care_vals).fillna(NAN_DUMMY_VALUE)
for i in range(10):
    if f"group/v{i}" in runs_df.columns:
        runs_df = runs_df.drop(columns=[f"group/v{i}"])

reference_runs = runs_df.groupby('model/name').head(1)

all_value_columns = list(runs_df.columns.drop('id'))
groups_mask = runs_df.duplicated(subset=all_value_columns, keep=False)
grouped_df = runs_df.groupby(all_value_columns, dropna=False)  # dropna=False is important! -> empty DF otherwise


def name_hparam_diff(diff: pd.Series):
    diff_to_ref_str = ""
    for k, v in diff.items():
        k = str(k)
        if k.split('/')[-1] in ['_target_', 'name']:
            k = ""  # k.replace('/_target_', '').replace('/name', '')  # optim/_target_ --> optim --> ""
        else:
            k = k.split('/')[-1]  # optim/lr --> lr
        # print(k, v)
        k = k.replace('hidden_dims', 'hdim').replace('kernels', 'kernel').replace('num_layers', 'L')
        k = k.replace("net_normalization", 'net_norm')
        k = k.replace('input_normalization', 'in-norm').replace('output_normalization', 'out-norm')
        k = k.replace('loss_function', 'loss')
        k = k.replace('num_', '#')
        k = k.replace('_', '-')
        if isinstance(v, tuple):
            v = 'x'.join([str(x) for x in v])  # (10, 128, 1) -> 10x128x1
        elif 'spatial_normalization_out' in k:
            k, v = '', 'SpatialNormOut'
        else:
            v = str(v).replace('False', 'NO').replace('none', 'No').replace('None', 'No')
            v = v.upper()
        diff_to_ref_str += f"{v}{k}_"
    return diff_to_ref_str.rstrip('_')


print('Reference runs:\n', reference_runs[['id', 'model/name', 'model/hidden_dims']])
for i, (name, group) in enumerate(grouped_df):
    group_IDs = list(group.id)

    model_name = group.iloc[0]['model/name']
    ref_run_series = reference_runs.loc[reference_runs['model/name'] == model_name].iloc[0]
    model_ref_run_id = ref_run_series['id']
    print(model_ref_run_id, group_IDs)
    if model_ref_run_id in group_IDs:
        diff_to_ref_str = f"REF_{model_name}"
        print(f'Ref run group {model_name}: ------------------>', group_IDs)
    else:
        diff = group.iloc[0][group.iloc[0] != ref_run_series].drop('id')
        diff_to_ref_str = name_hparam_diff(diff) + f'_{model_name}'
        diff_other = ref_run_series[group.iloc[0] != ref_run_series].drop('id')
        # print(f"Group vs ref:", pd.merge(group.iloc[0], ref_run_series, left_index=True, right_index=True))
        # print(f'Diff to ref run group {model_name}: ------------------>\n', diff_to_ref_str)

    any_group_run = api.run(f"{prefix}/{group.iloc[0]['id']}")
    hparas = {k:v for k, v in any_group_run.config.items() if 'model/' in k}
    other_esm_runs = filter_wandb_runs(hyperparam_filter=hparas, wandb_api=api)
    print(f'Other esm runs: {len(other_esm_runs)} for hparams {hparas}')
    for run in other_esm_runs:
        if f"group/v{version}" in run.config:
            continue
        print(model_name, run.id)
        run.config[f"group/v{version}"] = diff_to_ref_str
        try:
            run.update()
        except wandb.errors.CommError:
            logging.warning('---------------------------> FAILED:', run.id, '\n', '---' * 20)

'''
    hparas = {}
    for c in group.columns[group.columns.str.startswith('model/')]:
        if group.iloc[0][c] != NAN_DUMMY_VALUE:
            hparas[c] = group.iloc[0][c]
'''
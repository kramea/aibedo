import json
import logging
import os
import pathlib
from os.path import isdir, isfile
from typing import Union, Callable, List, Optional, Sequence

import wandb
import pandas as pd
from omegaconf import OmegaConf, DictConfig

from aibedo.utilities.config_utils import get_config_from_hydra_compose_overrides
from aibedo.utilities.utils import rsetattr, get_logger, get_local_ckpt_path, rhasattr, rgetattr

DF_MAPPING = Callable[[pd.DataFrame], pd.DataFrame]
log = get_logger(__name__)


def get_wandb_ckpt_name(run_path: str, epoch: Optional[int] = None) -> str:
    """
    Get the wandb ckpt name for a given run_path and epoch.
    Args:
        run_path: PROJECT/ENTITY/RUN_ID
        epoch: If a int, the ckpt name will be the one for that epoch, otherwise the latest epoch ckpt will be returned.

    Returns:
        The wandb ckpt file-name, that can be used as follows to restore the checkpoint locally:
           >>> ckpt_name = get_wandb_ckpt_name(run_path, epoch)
           >>> wandb.restore(ckpt_name, run_path=run_path, replace=True, root=os.getcwd())
    """
    run_api = wandb.Api(timeout=77).run(run_path)
    if 'best_model_filepath' in run_api.summary and epoch is None:
        best_model_path = run_api.summary['best_model_filepath']
    else:
        ckpt_files = [f.name for f in run_api.files() if f.name.endswith(".ckpt")]
        if len(ckpt_files) == 0:
            raise ValueError(f"Wandb run {run_path} has no checkpoint files (.ckpt) saved in the cloud!")
        elif len(ckpt_files) >= 2:
            ckpt_epochs = [int(name.replace('epoch', '')[:3]) for name in ckpt_files]
            if epoch is None:
                # Use checkpoint with latest epoch if epoch is not specified
                max_epoch = max(ckpt_epochs)
                best_model_path = [name for name in ckpt_files if str(max_epoch) in name][0]
                log.warning(f"Multiple ckpt files exist: {ckpt_files}. Using latest epoch: {best_model_path}")
            else:
                # Use checkpoint with specified epoch
                best_model_path = [name for name in ckpt_files if str(epoch) in name]
                if len(best_model_path) == 0:
                    raise ValueError(f"There is no ckpt file for epoch={epoch}. Try one of the ones in {ckpt_epochs}!")
                best_model_path = best_model_path[0]
        else:
            best_model_path = ckpt_files[0]
    return best_model_path


def restore_model_from_wandb_cloud(run_path: str) -> str:
    """
    Restore the model from the wandb cloud to local file-system.
    Args:
        run_path: PROJECT/ENTITY/RUN_ID

    Returns:
        The ckpt filename that can be used to reload the model locally.
    """
    best_model_path = get_wandb_ckpt_name(run_path)
    best_model_path = best_model_path.split('/')[-1]
    wandb.restore(best_model_path, run_path=run_path, replace=True, root=os.getcwd())
    return best_model_path


def load_hydra_config_from_wandb(
        run_path: str,
        override_key_value: List[str] = None,
        try_local_recovery: bool = True,
) -> DictConfig:
    """
    Args:
        run_path (str): the wandb ENTITY/PROJECT/ID (e.g. ID=2r0l33yc) corresponding to the config to-be-reloaded
        override_key_value: each element is expected to have a "=" in it, like datamodule.num_workers=8
    """
    run = wandb.Api(timeout=77).run(run_path)
    if not isinstance(override_key_value, list):
        raise ValueError(f"override_key_value must be a list of strings, but has type {type(override_key_value)}")
    overrides = override_key_value
    # Download from wandb cloud
    wandb_restore_kwargs = dict(run_path=run_path, replace=True, root=os.getcwd())
    try:
        wandb.restore("hydra_config.yaml", **wandb_restore_kwargs)
        kwargs = dict(config_path='../../', config_name="hydra_config.yaml")
    except ValueError:  # hydra_config has not been saved to wandb :(
        overrides += json.load(wandb.restore("wandb-metadata.json", **wandb_restore_kwargs))['args']
        kwargs = dict()
        if len(overrides) == 0:
            raise ValueError("wandb-metadata.json had no args, are you sure this is correct?")
            # also wandb-metadata.json is unexpected (was likely overwritten)
    overrides += [f'logger.wandb.id={run.id}',
                  f'logger.wandb.entity={run.entity}',
                  f'logger.wandb.project={run.project}', f'logger.wandb.group={run.group}']
    config = get_config_from_hydra_compose_overrides(overrides, **kwargs)
    os.remove('hydra_config.yaml') if os.path.exists('hydra_config.yaml') else None
    if run.id != config.logger.wandb.id and run.id in config.logger.wandb.name:
        config.logger.wandb.id = run.id
    assert config.logger.wandb.id == run.id, f"{config.logger.wandb.id} != {run.id}. \nFull Hydra config: {config}"
    return config


def reload_checkpoint_from_wandb(run_id: str,
                                 entity: str = 'salv47',
                                 project: str = 'AIBEDO',
                                 epoch: Optional[int] = None,
                                 override_key_value: Union[Sequence[str], dict] = None,
                                 try_local_recovery: bool = True,
                                 ) -> dict:
    """
    Reload model checkpoint based on only the Wandb run ID

    Args:
        run_id (str): the wandb run ID (e.g. 2r0l33yc) corresponding to the model to-be-reloaded
        entity (str): the wandb entity corresponding to the model to-be-reloaded
        project (str): the project entity corresponding to the model to-be-reloaded
        epoch (None or int): If None, the reloaded model will be the best one stored (or the latest one stored),
                             if an int, the reloaded model will be the one save at that epoch (if it was saved, otherwise an error is thrown)
        override_key_value: If a dict, every k, v pair is used to override the reloaded (hydra) config,
                            e.g. pass {datamodule.num_workers: 8} to change the corresponding flag in config.
                            If a sequence, each element is expected to have a "=" in it, like datamodule.num_workers=8
        try_local_recovery: If True, try to reload the model from local file-system if it exists.
    """
    from aibedo.interface import reload_model_from_config_and_ckpt
    run_path = f"{entity}/{project}/{run_id}"
    config = load_hydra_config_from_wandb(run_path, override_key_value, try_local_recovery=try_local_recovery)
    OmegaConf.update(config, f'model.input_transform._target_',
                     str(rgetattr(config, f'model.input_transform._target_')).replace('aibedo_salva', 'aibedo'))
    for k in ['model', 'datamodule', 'model.mixer', 'model.input_transform']:
        if config.get(k):
            OmegaConf.update(config, f'{k}._target_',
                             str(rgetattr(config, f'{k}._target_')).replace('aibedo_salva', 'aibedo'))

    if try_local_recovery and get_local_ckpt_path(config, epoch=epoch) is not None:
        best_model_fname = best_model_path = get_local_ckpt_path(config, epoch=epoch)
        logging.info(f" Found a local ckpt for run {run_id} at {best_model_fname}, using it instead of wandb.")
    else:
        best_model_path = get_wandb_ckpt_name(run_path, epoch=epoch)
        best_model_fname = best_model_path.split('/')[-1]  # in case the file contains local dir structure
        # IMPORTANT ARGS replace=True: see https://github.com/wandb/client/issues/3247
        wandb_restore_kwargs = dict(run_path=run_path, replace=True, root=os.getcwd())
        wandb.restore(best_model_fname, **wandb_restore_kwargs)  # download from the cloud

    assert config.logger.wandb.id == run_id, f"{config.logger.wandb.id} != {run_id}"

    try:
        model = reload_model_from_config_and_ckpt(config, best_model_fname, load_datamodule=True)
    except RuntimeError as e:
        # print_config(config.model, fields='all')
        raise RuntimeError(f"You have probably changed the model code, making it incompatible with older model "
                           f"versions. Tried to reload the model ckpt for run.id={run_id} from {best_model_path}.\n"
                           f"config.model={config.model}\n{e}")
    return {'model': model[0], 'datamodule': model[1], 'config': config}


def reupload_run_history(run):
    """
    This function can be called when for weird reasons your logged metrics do not appear in run.summary.
    All metrics for each epoch (assumes that a key epoch=i for each epoch i was logged jointly with the metrics),
    will be reuploaded to the wandb run summary.
    """
    summary = {}
    for row in run.scan_history():
        if 'epoch' not in row.keys() or any(['gradients/' in k for k in row.keys()]):
            continue
        summary.update(row)
    run.summary.update(summary)


#####################################################################
#
# Pre-filtering of wandb runs
#
def has_finished(run):
    return run.state == "finished"


def has_final_metric(run) -> bool:
    return 'test/MERRA2/mse_epoch' in run.summary.keys() and 'test/ERA5/mse_epoch' in run.summary.keys()


def has_keys(keys: Union[str, List[str]]) -> Callable:
    if isinstance(keys, str):
        keys = [keys]
    return lambda run: all([(k in run.summary.keys() or k in run.config.keys()) for k in keys])


def has_max_metric_value(metric: str = 'test/MERRA2/mse_epoch', max_metric_value: float = 1.0) -> Callable:
    return lambda run: run.summary[metric] <= max_metric_value


def has_tags(tags: Union[str, List[str]]) -> Callable:
    if isinstance(tags, str):
        tags = [tags]
    return lambda run: all([tag in run.tags for tag in tags])


def hasnt_tags(tags: Union[str, List[str]]) -> Callable:
    if isinstance(tags, str):
        tags = [tags]
    return lambda run: all([tag not in run.tags for tag in tags])


def hyperparams_list_api(**hyperparams) -> dict:
    return [{f"config.{hyperparam.replace('.', '/')}": value for hyperparam, value in hyperparams.items()}]


def has_hyperparam_values(**hyperparams) -> Callable:
    return lambda run: all(hyperparam in run.config and run.config[hyperparam] == value
                           for hyperparam, value in hyperparams.items())


def larger_than(**kwargs) -> Callable:
    return lambda run: all(hasattr(run.config, hyperparam) and value > run.config[hyperparam]
                           for hyperparam, value in kwargs.items())


def lower_than(**kwargs) -> Callable:
    return lambda run: all(hasattr(run.config, hyperparam) and value < run.config[hyperparam]
                           for hyperparam, value in kwargs.items())


def df_larger_than(**kwargs) -> DF_MAPPING:
    def f(df) -> pd.DataFrame:
        for k, v in kwargs.items():
            df = df.loc[getattr(df, k) > v]
        return df

    return f


def df_lower_than(**kwargs) -> DF_MAPPING:
    def f(df) -> pd.DataFrame:
        for k, v in kwargs.items():
            df = df.loc[getattr(df, k) < v]
        return df

    return f


str_to_run_pre_filter = {
    'has_finished': has_finished,
    'has_final_metric': has_final_metric
}


#####################################################################
#
# Post-filtering of wandb runs (usually when you need to compare runs)
#
def topk_runs(k: int = 5,
              metric: str = 'val/mse_epoch',
              lower_is_better: bool = True) -> DF_MAPPING:
    if lower_is_better:
        return lambda df: df.nsmallest(k, metric)
    else:
        return lambda df: df.nlargest(k, metric)


def topk_run_of_each_model_type(k: int = 1,
                                metric: str = 'val/mse_epoch',
                                lower_is_better: bool = True) -> DF_MAPPING:
    topk_filter = topk_runs(k, metric, lower_is_better)

    def topk_runs_per_model(df: pd.DataFrame) -> pd.DataFrame:
        models = df.model.unique()
        dfs = []
        for model in models:
            dfs += [topk_filter(df[df.model == model])]
        return pd.concat(dfs)

    return topk_runs_per_model


def non_unique_cols_dropper(df: pd.DataFrame) -> pd.DataFrame:
    nunique = df.nunique()
    cols_to_drop = nunique[nunique == 1].index
    df = df.drop(cols_to_drop, axis=1)
    return df


def groupby(df: pd.DataFrame,
            group_by: Union[str, List[str]] = 'seed',
            metrics: List[str] = 'val/mse_epoch',
            keep_columns: List[str] = 'model/name') -> pd.DataFrame:
    """
    Args:
        df: pandas DataFrame to be grouped
        group_by: str or list of str defining the columns to group by
        metrics: list of metrics to compute the group mean and std over
        keep_columns: list of columns to keep in the resulting grouped DataFrame
    Returns:
        A dataframe grouped by `group_by` with columns
        `metric`/mean and `metric`/std for each metric passed in `metrics` and all columns in `keep_columns` remain intact.
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    if isinstance(keep_columns, str):
        keep_columns = [keep_columns]
    if isinstance(group_by, str):
        group_by = [group_by]

    grouped_df = df.groupby(group_by, as_index=False)
    agg_metrics = {m: ['mean', 'std'] for m in metrics}
    agg_remain_intact = {c: 'first' for c in keep_columns}
    # cols = [group_by] + keep_columns + metrics + ['id']
    # print(df[cols].isna().any())
    stats = grouped_df.agg({**agg_metrics, **agg_remain_intact})
    stats.columns = [(f'{c[0]}/{c[1]}' if c[1] in ['mean', 'std'] else c[0]) for c in stats.columns]
    for m in metrics:
        stats[f'{m}/std'].fillna(value=0, inplace=True)

    return stats


str_to_run_post_filter = {
    **{
        f"top{k}": topk_runs(k=k)
        for k in range(1, 21)
    },
    'best_per_model': topk_run_of_each_model_type(k=1),
    **{
        f'top{k}_per_model': topk_run_of_each_model_type(k=k)
        for k in range(1, 6)
    },
    'unique_columns': non_unique_cols_dropper,
}


def get_wandb_filters_dict_list_from_list(filters_list) -> dict:
    if filters_list is None:
        filters_list = []
    elif not isinstance(filters_list, list):
        filters_list: List[Union[Callable, str]] = [filters_list]
    filters_wandb = []  # dict()
    for f in filters_list:
        if isinstance(f, str):
            f = str_to_run_pre_filter[f.lower()]
        filters_wandb.append(f)
        # filters_wandb = {**filters_wandb, **f}
    return filters_wandb


def get_best_model_config(
        metric: str = 'val/mse_epoch',
        mode: str = 'min',
        filters: Union[str, List[Union[Callable, str]]] = 'has_finished',
        entity: str = "salv47",
        project: str = 'AIBEDO',
        wandb_api=None
) -> dict:
    filters_wandb = get_wandb_filters_dict_list_from_list(filters)
    api = wandb_api or wandb.Api(timeout=77)
    # Project is specified by <entity/project-name>
    pm = '+' if mode == 'min' else '-'
    print(filters_wandb)
    filters_wandb = {"$and": [filters_wandb]}
    runs = api.runs(f"{entity}/{project}", filters=filters_wandb, order=f'{pm}summary_metrics.{metric}')
    return {'id': runs[0].id, **runs[0].config}


def get_run_ids_for_hyperparams(hyperparams: dict,
                                **kwargs) -> List[str]:
    runs = filter_wandb_runs(hyperparams, **kwargs)
    run_ids = [run.id for run in runs]
    return run_ids


def filter_wandb_runs(hyperparam_filter: dict = None,
                      filter_functions: Sequence[Callable] = None,
                      order='-created_at',
                      entity: str = "salv47",
                      project: str = 'AIBEDO',
                      wandb_api=None,
                      verbose: bool = True
                      ):
    """
    Args:
        hyperparam_filter: a dict str -> value, e.g. {'model/name': 'mlp', 'datamodule/exp_type': 'pristine'}
        filter_functions: A set of callable functions that take a wandb run and return a boolean (True/False) so that
                            any run with one or more return values being False is discarded/filtered out
    """
    hyperparam_filter = hyperparam_filter or dict()
    filter_functions = filter_functions or []
    api = wandb_api or wandb.Api(timeout=100)
    filter_wandb_api, filters_post = dict(), dict()
    for k, v in hyperparam_filter.items():
        if any(tpl in k for tpl in ['datamodule', 'normalizer']):
            filter_wandb_api[k] = v
        else:
            filters_post[k.replace('.', '/')] = v  # wandb keys are / separated
    filter_wandb_api = hyperparams_list_api(**filter_wandb_api)
    filter_wandb_api = {"$and": filter_wandb_api}  # MongoDB query lang
    runs = api.runs(f"{entity}/{project}", filters=filter_wandb_api, per_page=100, order=order)
    n_runs1 = len(runs)
    filters_post_func = has_hyperparam_values(**filters_post)
    runs = [run for run in runs if filters_post_func(run) and all(f(run) for f in filter_functions)]
    if verbose:
        log.info(f"#Filtered runs = {len(runs)}, (wandb API filtered {n_runs1})")
    return runs


def get_runs_df(
        get_metrics: bool = True,
        hyperparam_filter: dict = None,
        run_pre_filters: Union[str, List[Union[Callable, str]]] = 'has_finished',
        run_post_filters: Union[str, List[Union[DF_MAPPING, str]]] = None,
        verbose: int = 1,
        make_hashable_df: bool = False,
        **kwargs
) -> pd.DataFrame:
    """

        get_metrics:
        run_pre_filters:
        run_post_filters:
        verbose: 0, 1, or 2, where 0 = no output at all, 1 is a bit verbose
    """
    if isinstance(run_pre_filters, str):
        run_pre_filters = [run_pre_filters]
    run_pre_filters = [(f if callable(f) else str_to_run_pre_filter[f.lower()]) for f in run_pre_filters]

    if run_post_filters is None:
        run_post_filters = []
    elif not isinstance(run_post_filters, list):
        run_post_filters: List[Union[Callable, str]] = [run_post_filters]
    run_post_filters = [(f if callable(f) else str_to_run_post_filter[f.lower()]) for f in run_post_filters]

    # Project is specified by <entity/project-name>
    runs = filter_wandb_runs(hyperparam_filter, run_pre_filters, **kwargs)
    summary_list = []
    config_list = []
    group_list = []
    name_list = []
    tag_list = []
    id_list = []
    for i, run in enumerate(runs):
        if i % 50 == 0:
            print(f"Going after run {i}")
        # if i > 100: break
        # run.summary are the output key/values like accuracy.
        # We call ._json_dict to omit large files
        if 'model/_target_' not in run.config.keys():
            if verbose >= 1:
                print(f"Run {run.name} filtered out, as model/_target_ not in run.config.")
            continue

        id_list.append(str(run.id))
        tag_list.append(str(run.tags))
        if get_metrics:
            summary_list.append(run.summary._json_dict)
            # run.config is the input metrics.
            config_list.append({k: v for k, v in run.config.items() if k not in run.summary.keys()})
        else:
            config_list.append(run.config)

        # run.name is the name of the run.
        name_list.append(run.name)
        group_list.append(run.group)

    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    name_df = pd.DataFrame({'name': name_list, 'id': id_list, 'tags': tag_list})
    group_df = pd.DataFrame({'group': group_list})
    all_df = pd.concat([name_df, config_df, summary_df, group_df], axis=1)

    cols = [c for c in all_df.columns if not c.startswith('gradients/') and c != 'graph_0']
    all_df = all_df[cols]
    if all_df.empty:
        raise ValueError('Empty DF!')
    for post_filter in run_post_filters:
        all_df = post_filter(all_df)
    if make_hashable_df:
        all_df = all_df.applymap(lambda x: tuple(x) if isinstance(x, list) else x)

    all_df = clean_hparams(all_df)
    return all_df


def clean_hparams(df: pd.DataFrame):
    # Combine/unify columns of optim/scheduler which might be present in stored params more than once
    combine_cols = [col for col in df.columns if col.startswith('model/optim') or col.startswith('model/scheduler')]
    for col in combine_cols:
        new_col = col.replace('model/', '').replace('optimizer', 'optim')
        if not hasattr(df, new_col):
            continue
        getattr(df, new_col).fillna(getattr(df, col), inplace=True)
        # E.g.: all_df.Temp_Rating.fillna(all_df.Farheit, inplace=True)
        del df[col]
    if 'model/time_length' in df.columns:
        getattr(df, 'model/time_length').fillna(4.0, inplace=True)

    return df

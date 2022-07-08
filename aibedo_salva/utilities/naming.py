from typing import Union, List
from omegaconf import DictConfig


def _shared_prefix(config: DictConfig, init_prefix: str = "") -> str:
    s = init_prefix if isinstance(init_prefix, str) else ""
    kwargs = dict(mixer=config.model.mixer._target_) if config.model.get('mixer') else dict()
    s += clean_name(config.model._target_, **kwargs)
    if config.get('normalizer') and config.normalizer.get('input_normalization'):
        s += f"_{config.normalizer.get('input_normalization').upper()}"
    return s.lstrip('_')


def get_name_for_hydra_config_class(config: DictConfig) -> str:
    if 'name' in config and config.get('name') is not None:
        return config.get('name')
    elif '_target_' in config:
        return config._target_.split('.')[-1]
    return "$"


def get_detailed_name(config) -> str:
    """ This is a prefix for naming the runs for a more agreeable logging."""
    s = config.get("name")
    s = _shared_prefix(config, init_prefix=s) + '_'

    hdims = config.model.get('hidden_dims')
    if hdims is None:
        num_L = config.model.get('num_layers') or config.model.get('depth')
        if config.model.get('hidden_dim') is not None:
            hdims = f"{config.model.get('hidden_dim')}x{num_L}"
    elif all([h == hdims[0] for h in hdims]):
        hdims = f"{hdims[0]}x{len(hdims)}"
    else:
        hdims = str(hdims)
    s += f"{hdims}h"
    s += f"{config.model.optimizer.get('lr')}lr_"
    if config.model.optimizer.get('weight_decay') and config.model.optimizer.get('weight_decay') > 0:
        s += f"{config.model.optimizer.get('weight_decay')}wd_"

    s += f"{config.get('seed')}seed"
    return s.replace('None', '')


def clean_name(class_name, **kwargs) -> str:
    if "AFNONet" in class_name:
        if 'mixer' not in kwargs or "AFNO1D_Mixing" in kwargs['mixer']:
            s = 'FNO'
        elif "SelfAttention" in kwargs['mixer']:
            s = 'self-attention'
        else:
            raise ValueError(class_name, kwargs)
    elif "AIBEDO_MLP" in class_name:
        s = 'MLP'
    elif "graph_network" in class_name:
        s = 'GraphNet'
    elif "CNN_Net" in class_name:
        s = 'CNN'
    elif "SphericalUNetLSTM" in class_name:
        s = 'SUNet2'
    elif "SphericalUNet" in class_name:
        s = 'SUNet'
    else:
        raise ValueError(class_name, kwargs)
    return s


def get_group_name(config) -> str:
    s = get_name_for_hydra_config_class(config.model)
    s = _shared_prefix(config, init_prefix=s)
    if config.datamodule.get('input_filename'):
        s += f"_{config.datamodule.input_filename}"

    if config.get('normalizer'):
        if config.normalizer.get('spatial_normalization_in') and config.normalizer.get('spatial_normalization_out'):
            s += '+spatialNormed'
        elif config.normalizer.get('spatial_normalization_in'):
            s += '+spatialInNormed'
        elif config.normalizer.get('spatial_normalization_out'):
            s += '+spatialOutNormed'

    return s

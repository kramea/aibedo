import os
from typing import Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from aibedo_salva.data_transforms.normalization import Normalizer
from aibedo_salva.datamodules.abstract_datamodule import AIBEDO_DataModule
from aibedo_salva.models.base_model import BaseModel

"""
In this file you can find helper functions to avoid model/data loading and reloading boilerplate code
"""


def get_model(config: DictConfig, **kwargs) -> BaseModel:
    """
    Args:
        config (DictConfig): A OmegaConf config (e.g. produced by hydra <config>.yaml file parsing)
    Returns:
        The model that you can directly use to train with pytorch-lightning
    """
    if config.get('normalizer'):
        # This can be a bit redundant with get_datamodule (normalizer is instantiated twice), but it is better to be
        # sure that the output_normalizer is used by the model in cases where pytorch-lightning is not used.
        # By default if you use pytorch-lightning, the correct output_normalizer is passed to the model before training,
        # even without the below
        normalizer: Normalizer = hydra.utils.instantiate(
            config.normalizer, _recursive_=False,
            datamodule_config=config.datamodule,
        )
        kwargs['output_normalizer'] = normalizer.output_normalizer
    model: BaseModel = hydra.utils.instantiate(
        config.model, _recursive_=False,
        datamodule_config=config.datamodule,
        **kwargs
    )
    return model


def get_datamodule(config: DictConfig) -> AIBEDO_DataModule:
    """
    Args:
        config (DictConfig): A OmegaConf config (e.g. produced by hydra <config>.yaml file parsing)
    Returns:
        A datamodule that you can directly use to train pytorch-lightning models
    """
    # First we instantiate our normalization preprocesser, then our datamodule, and finally the model
    normalizer: Normalizer = hydra.utils.instantiate(
        config.normalizer,
        datamodule_config=config.datamodule,
        _recursive_=False
    )
    data_module: AIBEDO_DataModule = hydra.utils.instantiate(
        config.datamodule, _recursive_=False,
        normalizer=normalizer,
        model_config=config.model,
    )
    return data_module


def get_model_and_data(config: DictConfig) -> (BaseModel, AIBEDO_DataModule):
    """
    Args:
        config (DictConfig): A OmegaConf config (e.g. produced by hydra <config>.yaml file parsing)
    Returns:
        A tuple of (model, datamodule), that you can directly use to train with pytorch-lightning with:
            trainer = pl.Trainer(...)
            trainer.fit(model=model, datamodule=datamodule)
    """
    data_module = get_datamodule(config)
    model: BaseModel = hydra.utils.instantiate(
        config.model, _recursive_=False,
        datamodule_config=config.datamodule,
        output_normalizer=data_module.normalizer.output_normalizer if data_module.normalizer else None,
    )
    return model, data_module


def reload_model_from_config_and_ckpt(config: DictConfig,
                                      model_path: str,
                                      load_datamodule: bool = False,
                                      ):
    model, data_module = get_model_and_data(config)
    # Reload model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_state = torch.load(model_path, map_location=device)['state_dict']
    model.load_state_dict(model_state)
    if load_datamodule:
        return model, data_module
    return model



import os
from typing import Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from aibedo.data_transforms.normalization import Normalizer
from aibedo.datamodules.abstract_datamodule import AIBEDO_DataModule
from aibedo.models.base_model import BaseModel

"""
In this file you can find helper functions to avoid model/data loading and reloading boilerplate code
"""


def get_model(config: DictConfig, **kwargs) -> BaseModel:
    """
    Args:
        config (DictConfig): A OmegaConf config (e.g. produced by hydra <config>.yaml file parsing)
    Returns:
        The model that you can directly use to train with pytorch-lightning

    Examples:

    .. code-block:: python

        config_mlp = get_config_from_hydra_compose_overrides(overrides=['model=mlp'])
        mlp_model = get_model(config_mlp)
        random_mlp_input = torch.randn(1, 100)
        random_prediction = mlp_model(random_mlp_input)
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

    Examples:

    .. code-block:: python

        cfg = get_config_from_hydra_compose_overrides(overrides=['datamodule=icosahedron', 'datamodule.order=5'])
        ico_dm = get_datamodule(cfg)
        print(f"Icosahedron datamodule with order {ico_dm.order}")
    """
    # First we instantiate the normalization preprocesser, then the datamodule
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
        A tuple of (model, datamodule), that you can directly use to train with pytorch-lightning

    Examples:

    .. code-block:: python

        cfg = get_config_from_hydra_compose_overrides(overrides=['datamodule=icosahedron', 'model=mlp'])
        mlp_model, icosahedron_data = get_model_and_data(cfg)
        trainer = pl.Trainer(max_epochs=10, gpus=-1)
        trainer.fit(model=model, datamodule=icosahedron_data)

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
    """

    Args:
        config: The config to use to reload the model
        model_path (str): The path to the model checkpoint
        load_datamodule (bool): Whether to return the datamodule too

    Returns:
        A tuple of (model, datamodule) if load_datamodule is True, otherwise just the reloaded model

    Examples:

    .. code-block:: python

        # If you used wandb to save the model, you can use the following to reload it
        run_path = ENTITY/PROJECT/RUN_ID   # wandb run id (you can find it on the wandb URL after runs/, e.g. 1f5ehvll)
        config = load_hydra_config_from_wandb(run_path, override_kwargs=['datamodule.num_workers=4', 'trainer.gpus=-1'])

        model, datamodule = reload_model_from_config_and_ckpt(config, model_path, load_datamodule=True)
        trainer = hydra.utils.instantiate(config.trainer, _recursive_=False)
        trainer.test(model=model, datamodule=datamodule)

    """
    model, data_module = get_model_and_data(config)
    # Reload model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_state = torch.load(model_path, map_location=device)['state_dict']
    model.load_state_dict(model_state)
    if load_datamodule:
        return model, data_module
    return model



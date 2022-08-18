import os
from typing import Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from aibedo.datamodules.abstract_datamodule import AIBEDO_DataModule
from aibedo.models.base_model import BaseModel

"""
In this file you can find helper functions to avoid model/data loading and reloading boilerplate code
"""


def get_model(config: DictConfig, **kwargs) -> BaseModel:
    r"""Get the AIBEDO model, a subclass of :class:`~aibedo.models.base_model.BaseModel`, as defined by the key value pairs in ``config.model``.

    Args:
        config (DictConfig): A OmegaConf config (e.g. produced by hydra <config>.yaml file parsing)
        **kwargs: Any additional keyword arguments for the model class (overrides any key in config, if present)

    Returns:
        BaseModel:
            The model that you can directly use to train with pytorch-lightning

    Examples:

    .. code-block:: python

        from aibedo.utilities.config_utils import get_config_from_hydra_compose_overrides

        config_mlp = get_config_from_hydra_compose_overrides(overrides=['model=mlp'])
        mlp_model = get_model(config_mlp)

        # Get a prediction for a (B, S, C) shaped input
        random_mlp_input = torch.randn(1, 100, 5)
        random_prediction = mlp_model.predict(random_mlp_input)
    """
    model: BaseModel = hydra.utils.instantiate(
        config.model, _recursive_=False,
        datamodule_config=config.datamodule,
        **kwargs
    )
    return model


def get_datamodule(config: DictConfig) -> AIBEDO_DataModule:
    r"""Get the datamodule, as defined by the key value pairs in ``config.datamodule``. A datamodule defines the data-loading logic as well as data related (hyper-)parameters like the batch size, number of workers, etc.

    Args:
        config (DictConfig): A OmegaConf config (e.g. produced by hydra <config>.yaml file parsing)

    Returns:
        AIBEDO_DataModule:
            A datamodule that you can directly use to train pytorch-lightning models

    Examples:

    .. code-block:: python

        from aibedo.utilities.config_utils import get_config_from_hydra_compose_overrides

        cfg = get_config_from_hydra_compose_overrides(overrides=['datamodule=icosahedron', 'datamodule.order=5'])
        ico_dm = get_datamodule(cfg)
        print(f"Icosahedron datamodule with order {ico_dm.order}")
    """
    data_module: AIBEDO_DataModule = hydra.utils.instantiate(
        config.datamodule, _recursive_=False,
        model_config=config.model,
    )
    return data_module


def get_model_and_data(config: DictConfig) -> (BaseModel, AIBEDO_DataModule):
    r"""Get the model and datamodule. This is a convenience function that wraps around :meth:`get_model` and :meth:`get_datamodule`.

    Args:
        config (DictConfig): A OmegaConf config (e.g. produced by hydra <config>.yaml file parsing)

    Returns:
        (BaseModel, AIBEDO_DataModule): A tuple of (model, datamodule), that you can directly use to train with pytorch-lightning

    Examples:

    .. code-block:: python

        from aibedo.utilities.config_utils import get_config_from_hydra_compose_overrides

        cfg = get_config_from_hydra_compose_overrides(overrides=['datamodule=icosahedron', 'model=mlp'])
        mlp_model, icosahedron_data = get_model_and_data(cfg)

        # Use the data from datamodule (its ``train_dataloader()``), to train the model for 10 epochs
        trainer = pl.Trainer(max_epochs=10, gpus=-1)
        trainer.fit(model=model, datamodule=icosahedron_data)

    """
    data_module = get_datamodule(config)
    model: BaseModel = hydra.utils.instantiate(
        config.model, _recursive_=False,
        datamodule_config=config.datamodule
    )
    return model, data_module


def reload_model_from_config_and_ckpt(config: DictConfig,
                                      model_path: str,
                                      load_datamodule: bool = False,
                                      ):
    r"""Load a model as defined by ``config.model`` and reload its weights from ``model_path``.


    Args:
        config (DictConfig): The config to use to reload the model
        model_path (str): The path to the model checkpoint (its weights)
        load_datamodule (bool): Whether to return the datamodule too. Default: ``False``

    Returns:
        BaseModel: The reloaded model if load_datamodule is ``False``, otherwise a tuple of (reloaded-model, datamodule)

    Examples:

    .. code-block:: python

        # If you used wandb to save the model, you can use the following to reload it
        from aibedo.utilities.wandb_api import load_hydra_config_from_wandb

        run_path = ENTITY/PROJECT/RUN_ID   # wandb run id (you can find it on the wandb URL after runs/, e.g. 1f5ehvll)
        config = load_hydra_config_from_wandb(run_path, override_kwargs=['datamodule.num_workers=4', 'trainer.gpus=-1'])

        model, datamodule = reload_model_from_config_and_ckpt(config, model_path, load_datamodule=True)

        # Test the reloaded model
        trainer = hydra.utils.instantiate(config.trainer, _recursive_=False)
        trainer.test(model=model, datamodule=datamodule)

    """
    model, data_module = get_model_and_data(config)
    # Reload model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_state = torch.load(model_path, map_location=device)['state_dict']
    # Reload weights
    model.load_state_dict(model_state)
    if load_datamodule:
        return model, data_module
    return model



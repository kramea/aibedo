import logging
import os
from typing import Sequence

import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from hydra.utils import instantiate as hydra_instantiate

import aibedo_salva.utilities.config_utils as cfg_utils
from aibedo_salva.interface import reload_model_from_config_and_ckpt
from aibedo_salva.utilities.wandb_api import get_wandb_ckpt_name, load_hydra_config_from_wandb, \
    restore_model_from_wandb_cloud


def reload_and_test_model(run_id: str,
                          checkpoint_path: str = None,
                          config: DictConfig = None,
                          entity='salv47',
                          project='AIBEDO',
                          train_model: bool = False,
                          test_model: bool = True,
                          override_kwargs: Sequence[str] = None
                          ):
    """
    This function reloads a model from a checkpoint and trains and/or tests it.
        -> If train_model is True, it trains the model (i.e. can be used to resume training).
        -> If test_model is True, it tests the model (i.e. can be used to test a trained model).

    Args:
        run_id (str): Wandb run id
        checkpoint_path (str): An optional local ckpt path to load the weights from. If None, the best one on wandb will be used.
        config: An optional config to load the model and data from. If None, the config is loaded from Wandb.
        entity (str): Wandb entity
        project (str): Wandb project
        train_model (bool): Whether to train the model before (optional) testing. Default is False.
        test_model (bool): Whether to test the model. Default is True.
        override_kwargs: A list of strings (of the form "key=value") to override the given/reloaded config with.
    """
    run_path = f"{entity}/{project}/{run_id}"
    if checkpoint_path is not None:  # local loading
        if checkpoint_path.endswith('.ckpt'):
            best_model_path = checkpoint_path
        else:
            try:
                best_model_path = checkpoint_path + "/" + get_wandb_ckpt_name(run_path)
            except IndexError:
                import wandb
                saved_files = [f.name for f in wandb.Api().run(run_path).files()]
                logging.warning(
                    f"Run {run_id} does not have a saved ckpt in {checkpoint_path}. All saved files: {saved_files}")
                best_model_path = restore_model_from_wandb_cloud(run_path)
    else:
        best_model_path = restore_model_from_wandb_cloud(run_path)

    hydra_composed = False
    if config is None:
        config = load_hydra_config_from_wandb(run_path, override_kwargs)
    seed_everything(config.seed, workers=True)
    cfg_utils.extras(config)

    model, datamodule = reload_model_from_config_and_ckpt(config, best_model_path, load_datamodule=True)

    # Init Lightning callbacks and loggers
    callbacks = cfg_utils.get_all_instantiable_hydra_modules(config, 'callbacks')
    if hydra_composed:
        loggers = WandbLogger(entity=entity, project=project, id=run_id, resume="allow", reinit=True)
    else:
        loggers = cfg_utils.get_all_instantiable_hydra_modules(config, 'logger')

    # Init Lightning trainer
    trainer: pl.Trainer = hydra_instantiate(
        config.trainer, callbacks=callbacks, logger=loggers, _convert_="partial",
        resume_from_checkpoint=best_model_path
        # , deterministic=True
    )
    if train_model:
        trainer.fit(model=model, datamodule=datamodule)

    if test_model:  # Testing:
        trainer.test(datamodule=datamodule, model=model)

    if config.get('logger') and config.logger.get("wandb"):
        import wandb
        wandb.finish()

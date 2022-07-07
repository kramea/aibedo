import logging
import os
from typing import Sequence

import pytorch_lightning as pl
import wandb
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
                          override_kwargs: Sequence[str] = None
                          ):
    run_path = f"{entity}/{project}/{run_id}"
    if checkpoint_path is not None:  # local loading
        if checkpoint_path.endswith('.ckpt'):
            best_model_path = checkpoint_path
        else:
            try:
                best_model_path = checkpoint_path + "/" + get_wandb_ckpt_name(run_path)
            except IndexError:
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

    # Testing:
    trainer.test(datamodule=datamodule, model=model)
    wandb.finish()

import hydra
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

import aibedo_salva.utilities.config_utils as cfg_utils
from aibedo_salva.interface import get_model_and_data


def run_model(config: DictConfig):
    seed_everything(config.seed)
    cfg_utils.extras(config)

    if config.get("print_config"):  # pretty print config yaml -- requires rich to be installed
        print_fields = ("model", "datamodule", "normalizer", 'seed', 'work_dir')  # or "all"
        cfg_utils.print_config(config, fields=print_fields)

    model, datamodule = get_model_and_data(config)

    # Init Lightning callbacks and loggers
    callbacks = cfg_utils.get_all_instantiable_hydra_modules(config, 'callbacks')
    loggers = cfg_utils.get_all_instantiable_hydra_modules(config, 'logger')

    # Init Lightning trainer
    trainer: pl.Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=loggers,  # , deterministic=True
    )

    # Send some parameters from config to all lightning loggers
    cfg_utils.log_hyperparameters(config=config, model=model, data_module=datamodule, trainer=trainer,
                                  callbacks=callbacks)

    trainer.fit(model=model, datamodule=datamodule)

    cfg_utils.save_hydra_config_to_wandb(config)

    # Testing:
    if config.get("test_after_training"):
        trainer.test(datamodule=datamodule, ckpt_path='best')

    if config.get('logger') and config.logger.get("wandb"):
        import wandb
        wandb.finish()

    final_model = model.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        datamodule_config=config.datamodule,
        output_normalizer=datamodule.normalizer.output_normalizer if datamodule.normalizer else None,
    )

    return final_model

@hydra.main(config_path="configs/", config_name="main_config.yaml", version_base=None)
def main(config: DictConfig):
    return run_model(config)


if __name__ == "__main__":
    main()

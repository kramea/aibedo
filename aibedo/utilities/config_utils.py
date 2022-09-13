import os
import time
import warnings
from typing import Union, Sequence, List

import hydra
import wandb
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict

from aibedo.utilities.naming import get_group_name, get_detailed_name, clean_name
from aibedo.utilities.utils import get_logger, no_op

log = get_logger(__name__)


@rank_zero_only
def print_config(
        config,
        fields: Union[str, Sequence[str]] = (
                "datamodule",
                "model",
                "trainer",
                # "callbacks",
                # "logger",
                "seed",
        ),
        resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Credits go to: https://github.com/ashleve/lightning-hydra-template

    Args:
        config (ConfigDict): Configuration
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """
    import importlib
    if not importlib.util.find_spec("rich") or not importlib.util.find_spec("omegaconf"):
        # no pretty printing
        return
    import rich.syntax  # IMPORTANT to have, otherwise errors are thrown
    import rich.tree

    style = "dim"
    tree = rich.tree.Tree(":gear: CONFIG", style=style, guide_style=style)
    if isinstance(fields, str):
        if fields.lower() == 'all':
            fields = config.keys()
        else:
            fields = [fields]

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration
    - forcing multi-gpu friendly configuration
    - checking if config values are valid
    - init wandb if wandb logging is being used

    Credits go to: https://github.com/ashleve/lightning-hydra-template

    Modifies DictConfig in place.
    """
    log = get_logger()

    # Create working dir if it does not exist yet
    if config.get('work_dir'):
        os.makedirs(name=config.get("work_dir"), exist_ok=True)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # force multi-gpu friendly configuration if <config.trainer.accelerator=ddp>
    accelerator = config.trainer.get("accelerator")
    if accelerator in ["ddp", "ddp_spawn", "dp", "ddp2"]:
        log.info(f"Forcing ddp friendly configuration! <config.trainer.accelerator={accelerator}>")
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False

    # Set a short name for the model
    model_name = config.model.get('name')
    if model_name is None or model_name == '':
        model_class = config.model.get('_target_')
        mixer = config.model.mixer.get('_target_') if config.model.get('mixer') else None
        dm_type = config.datamodule.get('_target_')
        config.model.name = clean_name(model_class, mixer=mixer, dm_type=dm_type)

    USE_WANDB = "logger" in config.keys() and config.logger.get("wandb")
    if USE_WANDB:
        if not config.logger.wandb.get('id'):  # no wandb id has been assigned yet
            wandb_id = wandb.util.generate_id()
            config.logger.wandb.id = wandb_id
        else:
            log.info(f" This experiment config already has a wandb run ID = {config.logger.wandb.id}")
        if not config.logger.wandb.get('group'):  # no wandb group has been assigned yet
            group_name = get_group_name(config)
            config.logger.wandb.group = group_name if len(group_name) < 128 else group_name[:128]
        if not config.logger.wandb.get('name'):  # no wandb name has been assigned yet
            config.logger.wandb.name = get_detailed_name(config) + '_' + time.strftime(
                '%Hh%Mm_on_%b_%d') + '_' + config.logger.wandb.id
        else:
            # Append the model name to the wandb name
            config.logger.wandb.name += '_' + config.model.name

    check_config_values(config)

    # Init to wandb from rank 0 only in multi-gpu mode
    if USE_WANDB and int(os.environ.get('LOCAL_RANK', 0)) == 0 and os.environ.get('NODE_RANK', 0):
        # wandb_kwargs: dict = OmegaConf.to_container(config.logger.wandb, resolve=True)  # DictConfig -> simple dict
        wandb_kwargs = {
            k: config.logger.wandb.get(k) for k in ['id', 'project', 'entity', 'name', 'group',
                                                    'tags', 'notes', 'reinit', 'mode', 'resume']
        }
        wandb_kwargs['dir'] = config.logger.wandb.get('save_dir')
        try:
            wandb.init(**wandb_kwargs)
        except wandb.errors.UsageError as e:
            log.warning(" You need to login to wandb! Otherwise, choose a different/no logger with `logger=none`!")
            raise e
        log.info(f"Wandb kwargs: {wandb_kwargs}")
        save_hydra_config_to_wandb(config)


def check_config_values(config: DictConfig):
    """ Check if config values are valid. """
    with open_dict(config):
        if "net_normalization" in config.model.keys():
            if config.model.net_normalization is None:
                config.model.net_normalization = "none"
            config.model.net_normalization = config.model.net_normalization.lower()

        physics_lams1 = tuple(config.model.get('physics_loss_weights'))
        physics_lams2 = [config.model.get(f'lambda_physics{i}') for i in range(1, 6)]
        if any([(p is not None) for p in physics_lams2]):
            # Only one of the two should be specified, check that (physics_lams1 is by default all zero):
            if any(p > 0 for p in physics_lams1):
                raise ValueError(f'Only one of the two physics loss weights args should be specified: '
                                 f'``model.physics_loss_weights={physics_lams1}`` or'
                                 f'``model.lambda_physics={physics_lams2}´´ (lambda_physics1, ..., lambda_physics5)')
            else:
                nonneg_precip = physics_lams2[3]
                assert nonneg_precip in [None, True, False], f'lambda_physics4 should be either None, True, or False'
                physics_lams2[3] = 1.0 if nonneg_precip else 0.0
                config.model.physics_loss_weights = tuple([p or 0.0 for p in physics_lams2])

        if not config.datamodule.get('use_crel', default_value=True):
            config.datamodule.input_vars = [v for v in config.datamodule.input_vars if not str(v).startswith('crel_')]
        if not config.datamodule.get('use_crelSurf', default_value=True):
            config.datamodule.input_vars = [v for v in config.datamodule.input_vars if not v.startswith('crelSurf_')]
        if not config.datamodule.get('use_cresSurf', default_value=True):
            config.datamodule.input_vars = [v for v in config.datamodule.input_vars if not v.startswith('cresSurf_')]

        if config.get('logger') and config.logger.get("wandb"):
            if 'callbacks' in config and config.callbacks.get('model_checkpoint'):
                wandb_model_run_id = config.logger.wandb.get('id')
                d = config.callbacks.model_checkpoint.dirpath
                if wandb_model_run_id is not None:
                    # Save model checkpoints to special folder <ckpt-dir>/<wandb-run-id>/
                    new_dir = os.path.join(d, wandb_model_run_id)
                    config.callbacks.model_checkpoint.dirpath = new_dir
                    os.makedirs(new_dir, exist_ok=True)
                    log.info(f" Model checkpoints will be saved in: {new_dir}")
        else:
            if config.get('callbacks') and 'wandb' in config.callbacks:
                raise ValueError("You are trying to use wandb callbacks but you aren't using a wandb logger!")
            # log.warning("Model checkpoints will not be saved because you are not using wandb!")
            config.save_config_to_wandb = False


def get_all_instantiable_hydra_modules(config, module_name: str):
    from hydra.utils import instantiate as hydra_instantiate
    modules = []
    if module_name in config:
        for _, module_config in config[module_name].items():
            if module_config is not None and "_target_" in module_config:
                try:
                    modules.append(hydra_instantiate(module_config))
                except omegaconf.errors.InterpolationResolutionError as e:
                    log.warning(f" Hydra could not instantiate {module_config} for module_name={module_name}")
                    raise e
                except hydra.errors.InstantiationException:
                    log.warning(f" Hydra had trouble instantiating {module_config} for module_name={module_name}")
                    modules.append(hydra_instantiate(module_config, settings=wandb.Settings(start_method='fork')))

    return modules


@rank_zero_only
def log_hyperparameters(
        config,
        model: pl.LightningModule,
        data_module: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Credits go to: https://github.com/ashleve/lightning-hydra-template

    Additionally saves:
        - number of {total, trainable, non-trainable} model parameters
    """

    def copy_and_ignore_keys(dictionary, *keys_to_ignore):
        new_dict = dict()
        for k in dictionary.keys():
            if k not in keys_to_ignore:
                new_dict[k] = dictionary[k]
        return new_dict

    params = dict()
    if 'seed' in config:
        params['seed'] = config['seed']
    if 'model' in config:
        params['model'] = config['model']

    # Remove redundant keys or those that are not important to know after training -- feel free to edit this!
    params["datamodule"] = copy_and_ignore_keys(config["datamodule"], 'pin_memory', 'num_workers')
    params['model'] = copy_and_ignore_keys(config['model'], 'optimizer', 'scheduler')
    params["trainer"] = copy_and_ignore_keys(config["trainer"])
    # encoder, optims, and scheduler as separate top-level key
    params['optim'] = config['model']['optimizer']
    params['scheduler'] = config['model']['scheduler'] if 'scheduler' in config['model'] else None
    # Add a clean name for the model, for easier reading (e.g. aibedo.model.MLP.AIBEDO_MLP -> MLP)
    model_class = config.model.get('_target_')
    mixer = config.model.mixer.get('_target_') if config.model.get('mixer') else None
    params['model/name_id'] = clean_name(model_class, mixer=mixer)

    if "callbacks" in config:
        if 'model_checkpoint' in config['callbacks']:
            params["model_checkpoint"] = copy_and_ignore_keys(
                config["callbacks"]['model_checkpoint'], 'save_top_k'
            )

    # save number of model parameters
    params["model/params_total"] = sum(p.numel() for p in model.parameters())
    params["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    params["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    params['dirs/work_dir'] = config.get('work_dir')
    params['dirs/ckpt_dir'] = config.get('ckpt_dir')
    params['dirs/wandb_save_dir'] = config.logger.wandb.save_dir if (
            config.get('logger') and config.logger.get('wandb')) else None

    # send hparams to all loggers (if any logger is used)
    if trainer.logger is not None:
        log.info("Logging hyperparameters to the PyTorch Lightning loggers.")
        trainer.logger.log_hyperparams(params)

        # disable logging any more hyperparameters for all loggers
        # this is just a trick to prevent trainer from logging hparams of model,
        # since we already did that above
        trainer.logger.log_hyperparams = no_op


@rank_zero_only
def save_hydra_config_to_wandb(config: DictConfig):
    if config.get('save_config_to_wandb'):
        log.info(f"Hydra config will be saved to WandB as hydra_config.yaml and in wandb run_dir: {wandb.run.dir}")
        # files in wandb.run.dir folder get directly uploaded to wandb
        filepath = os.path.join(wandb.run.dir, "hydra_config.yaml")
        with open(filepath, "w") as fp:
            OmegaConf.save(config, f=fp.name, resolve=True)
        wandb.save("hydra_config.yaml")
    else:
        log.info("Hydra config will NOT be saved to WandB.")


def get_config_from_hydra_compose_overrides(overrides: List[str],
                                            config_path: str = "../configs",
                                            config_name: str = "main_config.yaml",
                                            ) -> DictConfig:
    """
    Function to get a Hydra config manually based on a default config file and a list of override strings.
    This is an alternative to using hydra.main(..) and the command-line for overriding the default config.

    Args:
        overrides: A list of strings of the form "key=value" to override the default config with.
        config_path: Relative path to the folder where the default config file is located.
        config_name: Name of the default config file (.yaml ending).

    Returns:
        The resulting config object based on the default config file and the overrides.

    Examples:

    .. code-block:: python

        config = get_config_from_hydra_compose_overrides(overrides=['model=mlp', 'model.optimizer.lr=0.001'])
        print(f"Lr={config.model.optimizer.lr}, MLP hidden_dims={config.model.hidden_dims}")
    """
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    overrides = list(set(overrides))
    if '-m' in overrides:
        overrides.remove('-m')  # if multiruns flags are mistakenly in overrides
    # Not true?!: log.info(f" Initializing Hydra from {os.path.abspath(config_path)}/{config_name}")
    hydra.initialize(config_path=config_path, version_base=None)
    try:
        config = hydra.compose(config_name=config_name, overrides=overrides)
    finally:
        GlobalHydra.instance().clear()  # always clean up global hydra
    return config


def get_model_from_hydra_compose_overrides(overrides: List[str]):
    """
    Function to get a torch model manually based on a default config file and a list of override strings.

    Args:
        overrides: A list of strings of the form "key=value" to override the default config with.

    Returns:
        The model instantiated from the resulting config.

    Examples:

    .. code-block:: python

        mlp_model = get_model_from_hydra_compose_overrides(overrides=['model=mlp'])
        random_mlp_input = torch.randn(1, 100)
        random_prediction = mlp_model(random_mlp_input)
    """
    from aibedo.interface import get_model
    cfg = get_config_from_hydra_compose_overrides(overrides)
    return get_model(cfg)

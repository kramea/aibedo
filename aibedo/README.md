# Using the code-base
**IMPORTANT NOTE:** 
All commands in this README assume that you are in the [root of the repository](../) (and need to be run from there)!

## Train a model
From the [repository root](../run.py), please run the following in the command line:    

    python run.py trainer.gpus=0 model=mlp logger=none callbacks=default

This will train an MLP on the CPU using some default callbacks and hyperparameters, but no logging.

## Important Training Arguments and Hyperparameters
- To run on CPU use ``trainer.gpus=0``, to use a single GPU use ``trainer.gpus=1``, etc.
- To override the data directory you can override the flag ``datamodule.data_dir=<data-dir>``, or see the [data README](../data/README.md) for more options.
- A random seed for reproducibility can be set with ``seed=<seed>`` (by default it is ``11``).

### Directories for logging and checkpoints
By default,
- the checkpoints (i.e. model weights) are saved in ``out_dir/checkpoints/``,
- any logs are saved in ``out_dir/logs/``.

To change the name of out_dir in both subdirs above, you may simply use the flag ``work_dir=YOUR-OUT-DIR``.
To only change the name of the checkpoints directory, you may use the flag ``ckpt_dir=YOUR-CHECKPOINTS-DIR``.
To only change the name of the logs directory, you may use the flag ``log_dir=YOUR-LOGS-DIR``.

### Data parameters and structure

#### Data Structures
Currently, we support the following data types:

##### Spherical data
Please use the flag ``datamodule=icosahedron`` to train on this spherical icosahedral data type.

By default training will be done on icosahedral data of order ``6``. 
To change this, use the flag ``datamodule.order=<order>``.

##### Euclidean data
Please use the flag ``datamodule=euclidean`` to train on this Euclidean/2D data type (i.e. on the normal latitude-longitude grid).

#### General data-specific parameters
Important data-specific parameters can be all found in the 
[configs/datamodule/base_data_config](configs/datamodule/base_data_config.yaml) file. 
In particular:
- ``datamodule.data_dir``: the directory where the data must be stored (see the [data README](../data/README.md) for more details).
- ``datamodule.time_lag``: the time lag to use for the data (i.e. the number of time steps to predict into the future). Default: ``0``
- ``datamodule.esm_for_training``: the ESM to use for training. Default: ``CESM2``. 
To use multiple ESMs, use the flag ``'datamodule.esm_for_training=[CESM2, GFDL-ESM4]'`` (or use any other list, but make sure to have the data downloaded in the data dir!).
To train on all ESMs, use ``datamodule.input_vars=all``
- ``datamodule.ensemble_ids``: which ID of the ESM ensemble to use for training.
Default: ``any``, which takes "r1i1p1f1" or any other id found in the data folder. To train on all ensemble members, use ``datamodule.ensemble_ids=all``.
- ``datamodule.input_vars``: the input variables 
- ``datamodule.output_vars``: the output/predicted variables
- ``datamodule.batch_size``: the batch size to use for training.
- ``datamodule.num_workers``: the number of workers to use for loading the data.

You can override any of these parameters by adding ``datamodule.<parameter>=<value>`` to the command line.

### ML model parameters and architecture

#### Define the architecture
To train a pre-defined model do ``model=<model_name>``, e.g. ``model=sunet``, ``model=mlp``, etc.,
    where [configs/model/](configs/model)<model_name>.yaml must be the configuration file for the respective model.

You can also override any model hyperparameter by adding ``model.<hyperparameter>=<value>`` to the command line.
E.g.:
- to change the number of layers and dimensions in an MLP you would use 
``model=mlp 'model.hidden_dims=[128, 128, 128]'`` (note that the parentheses are needed when the value is a list).
- to change the MLP ratio in the AFNO model you would use ``model=afno_spherical 'model.mlp_ratio=0.5'``.

#### General model-specific parameters
Important model-specific parameters can be all found in the 
[configs/model/_base_model_config](configs/model/_base_model_config.yaml) file. 
In particular:
- ``model.physics_loss_weights``: defines the weights for the five physics losses. 
By default no constraint is applied , i.e. it is ``model.physics_loss_weights=[0, 0, 0, 0, 0]``.
For example, if you want to use the fourth constraint (non-negative precipitation), you would use ``'model.physics_loss_weights=[0, 0, 0, 1, 0]'``.
- ``model.scheduler``: the scheduler to use for the learning rate. Default: Exponential decay with gamma=0.98
- ``model.window``: the number of time steps to use as input
(i.e. if window=3, three months are used to predict datamodule.time_lag months into the future). Default: ``1``.
- ``model.monitor``: the logged metric to track for early-stopping, model-checkpointing and LR-scheduling. Default: ``val/mse``.

### Hyperparameter optimization
Hyperparameter optimization is supported via the Optuna Sweeper.
Please read the instructions for setting it up and running experiments with Optuna in the
[Optuna configs README](configs/optuna/README.md).


### Wandb support
<details>
  <summary><b> Requirement </b></summary>
The following requires you to have a wandb (team) account, and you need to login with ``wandb login`` before you can use it.

</details>

- To log metrics to [wandb](https://wandb.ai/site) use ``logger=wandb``.
- To use some nice wandb specific callbacks in addition, use ``callbacks=wandb`` (e.g. save the best trained model to the wandb cloud).

## Tips

<details>
    <summary><b> hydra.errors.InstantiationException </b></summary>

The ``hydra.errors.InstantiationException`` itself is not very informative, 
so you need to look at the preceding exception(s) (i.e. scroll up) to see what went wrong.
</details>

<details>
    <summary><b> Overriding nested Hydra config groups </b></summary>

Nested config groups need to be overridden with a slash - not with a dot, since it would be interpreted as a string otherwise.
For example, if you want to change the filter in the AFNO transformer:
``python run.py model=afno model/mixer=self_attention``
And if you want to change the optimizer, you should run:
``python run.py  model=graphnet  optimizer@model.optimizer=SGD``
</details>

<details>
  <summary><b> Local configurations </b></summary>

You can easily use a local config file (that,e.g., overrides data dirs, working dir etc.), by putting such a yaml config 
in the [configs/local/](configs/local) subdirectory (Hydra searches for & uses by default the file configs/local/default.yaml, if it exists)
</details>

<details>
    <summary><b> Wandb </b></summary>

If you use Wandb, make sure to select the "Group first prefix" option in the panel/workspace settings of the web app inside the project (in the top right corner).
This will make it easier to browse through the logged metrics.
</details>





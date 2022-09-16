# Train a model
From the repository root, please run the following in the command line:    

    python run.py trainer.gpus=0 model=mlp logger=none callbacks=default

This will train an MLP on the CPU using some default callbacks and hyperparameters, but no logging.

### Training Arguments and Hyperparameters
- To run on CPU ``trainer.gpus=0``, to use a single GPU ``trainer.gpus=1``, etc.
- To train a specific model ``model=<model_name>``, e.g. ``model=sunet``, ``model=mlp``, etc.,
    where [configs/model/](configs/model)<model_name>.yaml is the configuration file for the model.

- To override the data directory you can override the flag ``datamodule.data_dir=<data-dir>``

### Hyperparameter optimization
Please install the hydra-optuna-wandb integration with
    
    cd aibedo/hydra_optuna_sweeper
    python setup.py install --user  # or pip install -e . --user

To tune the hyperparameters with, e.g., Optuna for any of the provided ML models please run the following command:
``python train.py --multirun optuna=<ml-model-name>``, where ``ml-model-name`` is the name of the ML model (e.g. ``mlp``).

To extend this example with your own ML model, please create a config file in [configs/optuna/](configs/optuna).
Note that the config files defining the swept hyperparameters (and their possible values/ranges) for each model
are located in [configs/optuna/params](configs/optuna/params).

**Note:** Wandb has a known bug when logging too much - 
something along the lines of ``OSError: [Errno 24] Too many open files``. If that happens (or to prevent it), 
you can increase the limit with the following command: ``ulimit -n 65536``.

### Wandb support
<details>
  <summary><b> Requirement </b></summary>
The following requires you to have a wandb (team) account, and you need to login with ``wandb login`` before you can use it.

</details>

- To log metrics to [wandb](https://wandb.ai/site) use ``logger=wandb``.
- To use some nice wandb specific callbacks in addition, use ``callbacks=wandb`` (e.g. save the best trained model to the wandb cloud).

## Tips

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

<details>
    <summary><b> Credits & Resources </b></summary>

The following template was extremely useful for getting started with the PL+Hydra implementation:
[ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
</details>





# Environment
    conda env create -f env.yaml   # create new environment will all dependencies
    conda activate aibedo_salva  # activate the environment called 'aibedo_salva'

# Train a model

    python train.py trainer.gpus=0 model=sunet logger=none callbacks=default

### Training Arguments and Hyperparameters
- To run on CPU ``trainer.gpus=0``, to use a single GPU ``trainer.gpus=1``, etc.
- To train a specific model ``model=<model_name>``, e.g. ``model=sunet``, ``model=mlp``, etc.,
    where [configs/model/](configs/model)<model_name>.yaml is the configuration file for the model.

- To override the data directory use either of the two options below:
   1) ``datamodule.data_dir=<data-dir>``
   2) Create a config file called [configs/local/](configs/local)default.yaml adapted 
  from e.g. this example local config [configs/local/example_local_config.yaml](configs/local/example_local_config.yaml).
  Its values will be automatically used whenever you run ``train.py``.
### Wandb support
<details>
  <summary><b> Requirement </b></summary>
The following requires you to have a wandb (team) account and you need to login with ``wandb login`` before you can use it.

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

If you use Wandb, make sure to select the "Group first prefix" option in the panel settings of the web app.
This will make it easier to browse through the logged metrics.
</details>

<details>
    <summary><b> Credits & Resources </b></summary>

The following template was extremely useful for getting started with the PL+Hydra implementation:
[ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
</details>





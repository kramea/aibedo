## Hyperparameter optimization

### Setting up Optuna
Make sure to have optuna installed (``pip install optuna``).

Then, please install the hydra-optuna-wandb integration with

    cd ../../hydra_optuna_sweeper
    pip install -e . --user

Note: this is the same as following the instructions in the [Optuna sweeper README](../../hydra_optuna_sweeper/README.md).

### Running a HPO experiment
**Note:** All following commands are to be ran from the [repository root](../../run.py).

To tune the hyperparameters with Optuna for any of the provided ML models please run the following command:

    python run.py --multirun optuna=HPO-EXP-NAME
 
where ``HPO-EXP-NAME`` is the name of the HPO experiment that you want to run (e.g. ``optuna=mlp1d``), 
as defined by a corresponding ``HPO-EXP-NAME.yaml`` file in this folder.

### Extending the Optuna Sweeper to other models/parameters
To extend this example with your own ML model, please create a corresponding config file in this folder (i.e. in aibedo/configs/optuna/).
Note that the config files defining the swept hyperparameters (and their possible values/ranges) for each model
are located in [aibedo/configs/optuna/params](./params).

### WandB integration
**Note:** Wandb has a known bug when logging too much - 
something along the lines of ``OSError: [Errno 24] Too many open files``. If that happens (or to prevent it), 
you can increase the limit with the following command: ``ulimit -n 65536``.


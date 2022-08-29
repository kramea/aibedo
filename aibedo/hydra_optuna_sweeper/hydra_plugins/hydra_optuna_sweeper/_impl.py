# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import functools
import logging
import os
import sys
import tempfile
import warnings
from copy import copy
from textwrap import dedent
from typing import (
    Any,
    Callable,
    Dict,
    List,
    MutableMapping,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
)

import optuna
from hydra._internal.deprecation_warning import deprecation_warning
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.override_parser.types import (
    ChoiceSweep,
    IntervalSweep,
    Override,
    RangeSweep,
    Transformer,
)
from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf
from optuna.distributions import (
    BaseDistribution,
    CategoricalChoiceType,
    CategoricalDistribution,
    DiscreteUniformDistribution,
    IntLogUniformDistribution,
    IntUniformDistribution,
    LogUniformDistribution,
    UniformDistribution,
)
from optuna.trial import Trial

from .config import Direction, DistributionConfig, DistributionType

log = logging.getLogger(__name__)


def create_optuna_distribution_from_config(
        config: MutableMapping[str, Any]
) -> BaseDistribution:
    kwargs = dict(config)
    if isinstance(config["type"], str):
        kwargs["type"] = DistributionType[config["type"]]
    param = DistributionConfig(**kwargs)
    if param.type == DistributionType.categorical:
        assert param.choices is not None
        return CategoricalDistribution(param.choices)
    if param.type == DistributionType.int:
        assert param.low is not None
        assert param.high is not None
        if param.log:
            return IntLogUniformDistribution(int(param.low), int(param.high))
        step = int(param.step) if param.step is not None else 1
        return IntUniformDistribution(int(param.low), int(param.high), step=step)
    if param.type == DistributionType.float:
        assert param.low is not None
        assert param.high is not None
        if param.log:
            return LogUniformDistribution(param.low, param.high)
        if param.step is not None:
            return DiscreteUniformDistribution(param.low, param.high, param.step)
        return UniformDistribution(param.low, param.high)
    raise NotImplementedError(f"{param.type} is not supported by Optuna sweeper.")


def create_optuna_distribution_from_override(override: Override) -> Any:
    if not override.is_sweep_override():
        return override.get_value_element_as_str()

    value = override.value()
    choices: List[CategoricalChoiceType] = []
    if override.is_choice_sweep():
        assert isinstance(value, ChoiceSweep)
        for x in override.sweep_iterator(transformer=Transformer.encode):
            assert isinstance(
                x, (str, int, float, bool, type(None))
            ), f"A choice sweep expects str, int, float, bool, or None type. Got {type(x)}."
            choices.append(x)
        return CategoricalDistribution(choices)

    if override.is_range_sweep():
        assert isinstance(value, RangeSweep)
        assert value.start is not None
        assert value.stop is not None
        if value.shuffle:
            for x in override.sweep_iterator(transformer=Transformer.encode):
                assert isinstance(
                    x, (str, int, float, bool, type(None))
                ), f"A choice sweep expects str, int, float, bool, or None type. Got {type(x)}."
                choices.append(x)
            return CategoricalDistribution(choices)
        if (
                isinstance(value.start, float)
                or isinstance(value.stop, float)
                or isinstance(value.step, float)
        ):
            return DiscreteUniformDistribution(value.start, value.stop, value.step)
        return IntUniformDistribution(
            int(value.start), int(value.stop), step=int(value.step)
        )

    if override.is_interval_sweep():
        assert isinstance(value, IntervalSweep)
        assert value.start is not None
        assert value.end is not None
        if "log" in value.tags:
            if isinstance(value.start, int) and isinstance(value.end, int):
                return IntLogUniformDistribution(int(value.start), int(value.end))
            return LogUniformDistribution(value.start, value.end)
        else:
            if isinstance(value.start, int) and isinstance(value.end, int):
                return IntUniformDistribution(value.start, value.end)
            return UniformDistribution(value.start, value.end)

    raise NotImplementedError(f"{override} is not supported by Optuna sweeper.")


def create_params_from_overrides(
        arguments: List[str],
) -> Tuple[Dict[str, BaseDistribution], Dict[str, Any]]:
    parser = OverridesParser.create()
    parsed = parser.parse_overrides(arguments)
    search_space_distributions = dict()
    fixed_params = dict()

    for override in parsed:
        param_name = override.get_key_element()
        value = create_optuna_distribution_from_override(override)
        if isinstance(value, BaseDistribution):
            search_space_distributions[param_name] = value
        else:
            fixed_params[param_name] = value
    return search_space_distributions, fixed_params


class OptunaSweeperImpl(Sweeper):
    def __init__(
            self,
            sampler: Any,
            direction: Any,
            storage: Optional[Any],
            study_name: Optional[str],
            n_trials: int,
            n_jobs: int,
            max_failure_rate: float,
            search_space: Optional[DictConfig],
            custom_search_space: Optional[str],
            params: Optional[DictConfig],
    ) -> None:
        self.sampler = sampler
        self.direction = direction
        self.storage = storage
        self.study_name = study_name
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.max_failure_rate = max_failure_rate
        assert self.max_failure_rate >= 0.0
        assert self.max_failure_rate <= 1.0
        self.custom_search_space_extender: Optional[
            Callable[[DictConfig, Trial], None]
        ] = None
        if custom_search_space:
            self.custom_search_space_extender = get_method(custom_search_space)
        self.search_space = search_space
        self.params = params
        self.job_idx: int = 0
        self.search_space_distributions: Optional[Dict[str, BaseDistribution]] = None

    def _process_searchspace_config(self) -> None:
        url = "https://hydra.cc/docs/upgrades/1.1_to_1.2/changes_to_sweeper_config/"
        if self.params is None and self.search_space is None:
            self.params = OmegaConf.create({})
        elif self.search_space is not None:
            if self.params is not None:
                warnings.warn(
                    "Both hydra.sweeper.params and hydra.sweeper.search_space are configured."
                    "\nHydra will use hydra.sweeper.params for defining search space."
                    f"\n{url}"
                )
            else:
                deprecation_warning(
                    message=dedent(
                        f"""\
                        `hydra.sweeper.search_space` is deprecated and will be removed in the next major release.
                        Please configure with `hydra.sweeper.params`.
                        {url}
                        """
                    ),
                )
                self.search_space_distributions = {
                    str(x): create_optuna_distribution_from_config(y)
                    for x, y in self.search_space.items()
                }

    def setup(
            self,
            *,
            hydra_context: HydraContext,
            task_function: TaskFunction,
            config: DictConfig,
    ) -> None:
        self.job_idx = 0
        self.config = config
        self.hydra_context = hydra_context
        self.launcher = Plugins.instance().instantiate_launcher(
            config=config, hydra_context=hydra_context, task_function=task_function
        )
        self.sweep_dir = config.hydra.sweep.dir

    def _get_directions(self) -> List[str]:
        if isinstance(self.direction, MutableSequence):
            return [d.name if isinstance(d, Direction) else d for d in self.direction]
        elif isinstance(self.direction, str):
            return [self.direction]
        return [self.direction.name]

    def _configure_trials(
            self,
            trials: List[Trial],
            search_space_distributions: Dict[str, BaseDistribution],
            fixed_params: Dict[str, Any],
    ) -> Sequence[Sequence[str]]:
        overrides = []
        for trial in trials:
            for param_name, distribution in search_space_distributions.items():
                assert type(param_name) is str
                trial._suggest(param_name, distribution)
            for param_name, value in fixed_params.items():
                trial.set_user_attr(param_name, value)

            if self.custom_search_space_extender:
                assert self.config is not None
                self.custom_search_space_extender(self.config, trial)

            overlap = trial.params.keys() & trial.user_attrs
            if len(overlap):
                raise ValueError(
                    "Overlapping fixed parameters and search space parameters found!"
                    f"Overlapping parameters: {list(overlap)}"
                )
            params = dict(trial.params)
            params.update(fixed_params)

            overrides.append(tuple(f"{name}={val}" for name, val in params.items()))
        return overrides

    def _parse_sweeper_params_config(self) -> List[str]:
        if not self.params:
            return []

        return [f"{k!s}={v}" for k, v in self.params.items()]

    def _to_grid_sampler_choices(self, distribution: BaseDistribution) -> Any:
        if isinstance(distribution, CategoricalDistribution):
            return distribution.choices
        elif isinstance(distribution, IntUniformDistribution):
            assert (
                    distribution.step is not None
            ), "`step` of IntUniformDistribution must be a positive integer."
            n_items = (distribution.high - distribution.low) // distribution.step
            return [distribution.low + i * distribution.step for i in range(n_items)]
        elif isinstance(distribution, DiscreteUniformDistribution):
            n_items = int((distribution.high - distribution.low) // distribution.q)
            return [distribution.low + i * distribution.q for i in range(n_items)]
        else:
            raise ValueError("GridSampler only supports discrete distributions.")

    def sweep(self, arguments: List[str]) -> None:
        assert self.config is not None
        assert self.launcher is not None
        assert self.hydra_context is not None
        assert self.job_idx is not None

        self._process_searchspace_config()
        params_conf = self._parse_sweeper_params_config()
        params_conf.extend(arguments)

        is_grid_sampler = (
                isinstance(self.sampler, functools.partial)
                and self.sampler.func == optuna.samplers.GridSampler  # type: ignore
        )

        (
            override_search_space_distributions,
            fixed_params,
        ) = create_params_from_overrides(params_conf)

        search_space_distributions = dict()
        if self.search_space_distributions:
            search_space_distributions = self.search_space_distributions.copy()
        search_space_distributions.update(override_search_space_distributions)

        if is_grid_sampler:
            search_space_for_grid_sampler = {
                name: self._to_grid_sampler_choices(distribution)
                for name, distribution in search_space_distributions.items()
            }

            self.sampler = self.sampler(search_space_for_grid_sampler)
            n_trial = 1
            for v in search_space_for_grid_sampler.values():
                n_trial *= len(v)
            self.n_trials = min(self.n_trials, n_trial)
            log.info(
                f"Updating num of trials to {self.n_trials} due to using GridSampler."
            )

        # Remove fixed parameters from Optuna search space.
        for param_name in fixed_params:
            if param_name in search_space_distributions:
                del search_space_distributions[param_name]

        directions = self._get_directions()

        def get_new_study():
            optuna_study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                sampler=self.sampler,
                directions=directions,
                load_if_exists=True,
            )
            return optuna_study

        if self.config.get('logger') and self.config.logger.get('wandb'):
            import wandb
            dummy_cfg = copy(self.config)
            dummy_cfg.hydra = None
            # print(OmegaConf.to_yaml(dummy_cfg, resolve=True))
            OmegaConf.resolve(dummy_cfg)   # resolve interpolations etc.
            # Get wandb info from config
            wandb_cfg = dummy_cfg.logger.wandb
            project = wandb_cfg.get('project')
            entity = wandb_cfg.get('entity')
            optional_resume_study_id = dummy_cfg.optuna.get('wandb_study_id')
            if optional_resume_study_id is None:
                study_run_id = wandb.util.generate_id()   # get a new wandb id
                study = get_new_study()
            else:
                study_run_id = optional_resume_study_id
                study = reload_study_from_wandb(run_path=f'{entity}/{project}/{study_run_id}')
            wandb.init(id=study_run_id,
                       name='Optuna-' + self.study_name,
                       tags=['optuna', self.study_name],
                       project=project,
                       entity=entity,
                       reinit=True, job_type='optuna_study')
            # some logging of info
            if optional_resume_study_id is None:
                log.info(f" Creating Optuna study with Wandb study run id: {study_run_id}")
            else:
                log.info(f" Loading & resuming Optuna study from Wandb study run id: {study_run_id}")
            wandb.finish()
        else:
            # No wandb involved whatsoever, just create a new study.
            study = get_new_study()
        log.info(f"Study name: {study.study_name}")
        log.info(f"Storage: {self.storage}")
        log.info(f"Sampler: {type(self.sampler).__name__}")
        log.info(f"Directions: {directions}")

        batch_size = self.n_jobs
        n_trials_to_go = self.n_trials

        while n_trials_to_go > 0:
            batch_size = min(n_trials_to_go, batch_size)

            trials = [study.ask() for _ in range(batch_size)]
            overrides = self._configure_trials(
                trials, search_space_distributions, fixed_params
            )

            returns = self.launcher.launch(overrides, initial_job_idx=self.job_idx)
            self.job_idx += len(returns)
            failures = []
            for trial, ret in zip(trials, returns):
                trial_cfg = ret.cfg
                trial_cfg_logger = trial_cfg["logger"]

                values: Optional[List[float]] = None
                state: optuna.trial.TrialState = optuna.trial.TrialState.COMPLETE
                try:
                    if len(directions) == 1:
                        try:
                            values = [float(ret.return_value)]
                        except (ValueError, TypeError):
                            raise ValueError(
                                f"Return value must be float-castable. Got '{ret.return_value}'."
                            ).with_traceback(sys.exc_info()[2])
                    else:
                        try:
                            values = [float(v) for v in ret.return_value]
                        except (ValueError, TypeError):
                            raise ValueError(
                                "Return value must be a list or tuple of float-castable values."
                                f" Got '{ret.return_value}'."
                            ).with_traceback(sys.exc_info()[2])
                        if len(values) != len(directions):
                            raise ValueError(
                                "The number of the values and the number of the objectives are"
                                f" mismatched. Expect {len(directions)}, but actually {len(values)}."
                            )
                    try:
                        study.tell(trial=trial, state=state, values=values)
                    except RuntimeError as e:
                        if (
                                is_grid_sampler
                                and "`Study.stop` is supposed to be invoked inside an objective function or a callback."
                                in str(e)
                        ):
                            pass
                        else:
                            raise e

                    if 'wandb' in trial_cfg_logger:
                        lcb = OptunaWandbCallback(trial_cfg_logger['wandb'], run_id=study_run_id)
                        lcb(study, trial)
                        lcb.end()

                except Exception as e:
                    state = optuna.trial.TrialState.FAIL
                    study.tell(trial=trial, state=state, values=values)
                    log.warning(f"Failed experiment: {e}")
                    failures.append(e)

            # raise if too many failures
            if len(failures) / len(returns) > self.max_failure_rate:
                log.error(
                    f"Failed {failures} times out of {len(returns)} "
                    f"with max_failure_rate={self.max_failure_rate}."
                )
                assert len(failures) > 0
                for ret in returns:
                    ret.return_value  # delegate raising to JobReturn, with actual traceback

            n_trials_to_go -= batch_size

        results_to_serialize: Dict[str, Any]
        if len(directions) < 2:
            best_trial = study.best_trial
            results_to_serialize = {
                "name": "optuna",
                "best_params": best_trial.params,
                "best_value": best_trial.value,
            }
            log.info(f"Best parameters: {best_trial.params}")
            log.info(f"Best value: {best_trial.value}")
        else:
            best_trials = study.best_trials
            pareto_front = [
                {"params": t.params, "values": t.values} for t in best_trials
            ]
            results_to_serialize = {
                "name": "optuna",
                "solutions": pareto_front,
            }
            log.info(f"Number of Pareto solutions: {len(best_trials)}")
            for t in best_trials:
                log.info(f"    Values: {t.values}, Params: {t.params}")
        OmegaConf.save(
            OmegaConf.create(results_to_serialize),
            f"{self.config.hydra.sweep.dir}/optimization_results.yaml",
        )


# ----------------------------------------------------------------------------------------------------------------------
import wandb
import joblib
from typing import Iterable, Union


class OptunaWandbCallback:
    """A callback that logs the metadata from Optuna Study to wandb.

    With this callback, you can log and display:

    * values and params for each trial
    * current best values and params for the study
    * visualizations from the `optuna.visualizations` module
    * parameter distributions for each trial
    * study object itself to load it later
    * and more

    Args:
        run: wandb Run.
        plots_update_freq(int, str, optional): Frequency at which plots are logged and updated in Neptune.
            If you pass integer value k, plots will be updated every k iterations.
            If you pass the string 'never', plots will not be logged. Defaults to 1.
        study_update_freq(int, str, optional): It is a frequency at which a study object is logged and updated in Neptune.
            If you pass integer value k, the study will be updated every k iterations.
            If you pass the string 'never', plots will not be logged. Defaults to 1.
        visualization_backend(str, optional): Which visualization backend is used for 'optuna.visualizations' plots.
            It can be one of 'matplotlib' or 'plotly'. Defaults to 'plotly'.
        log_plot_contour(bool, optional): If 'True' the `optuna.visualizations.plot_contour`
            visualization will be logged to Neptune. Defaults to `True`.
        log_plot_edf(bool, optional): If 'True' the `optuna.visualizations.plot_edf`
            visualization will be logged to Neptune. Defaults to `True`.
        log_plot_parallel_coordinate(bool, optional): If 'True' the `optuna.visualizations.plot_parallel_coordinate`
            visualization will be logged to Neptune. Defaults to `True`.
        log_plot_param_importances(bool, optional): If 'True' the `optuna.visualizations.plot_param_importances`
            visualization will be logged to Neptune. Defaults to `True`.
        log_plot_pareto_front(bool, optional): If 'True' the `optuna.visualizations.plot_pareto_front`
            visualization will be logged to Neptune.
            If your `optuna.study` is not multi-objective this plot is not logged. Defaults to `True`.
        log_plot_slice(bool, optional): If 'True' the `optuna.visualizations.plot_slice`
            visualization will be logged to Neptune. Defaults to `True`.
        log_plot_intermediate_values(bool, optional): If 'True' the `optuna.visualizations.plot_intermediate_values`
            visualization will be logged to Neptune.
            If your `optuna.study` is not using pruners this plot is not logged. Defaults to `True`. Defaults to `True`.
        log_plot_optimization_history(bool, optional): If 'True' the `optuna.visualizations.plot_optimization_history`
            visualization will be logged to Neptune. Defaults to `True`.

    """

    def __init__(self,
                 run_kwargs,
                 run_id: str,
                 plots_update_freq: Union[int, str] = 1,
                 study_update_freq: Union[int, str] = 1,
                 visualization_backend: str = 'plotly',
                 log_plot_contour: bool = True,
                 log_plot_edf: bool = True,
                 log_plot_parallel_coordinate: bool = True,
                 log_plot_param_importances: bool = True,
                 log_plot_pareto_front: bool = True,
                 log_plot_slice: bool = True,
                 log_plot_intermediate_values: bool = True,
                 log_plot_optimization_history: bool = True):

        self.run = wandb.init(project=run_kwargs['project'], id=run_id, reinit=True, resume=True)
        self._visualization_backend = visualization_backend
        self._plots_update_freq = plots_update_freq
        self._study_update_freq = study_update_freq
        self._log_plot_contour = log_plot_contour
        self._log_plot_edf = log_plot_edf
        self._log_plot_parallel_coordinate = log_plot_parallel_coordinate
        self._log_plot_param_importances = log_plot_param_importances
        self._log_plot_pareto_front = log_plot_pareto_front
        self._log_plot_slice = log_plot_slice
        self._log_plot_intermediate_values = log_plot_intermediate_values
        self._log_plot_optimization_history = log_plot_optimization_history

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        self._log_trial(trial)
        self._log_trial_distributions(trial)
        self._log_best_trials(study)
        self._log_study_details(study, trial)
        self._log_plots(study, trial)
        self._log_study(study, trial)

    def end(self):
        self.run.finish()

    def _log_trial(self, trial):
        if hasattr(trial, 'value'):
            logging.info(f"Trial {trial._trial_id}: value={trial.value}")
        pass
        # _log_trials(self.run, [trial])

    def _log_trial_distributions(self, trial):
        self.run.log({'study/distributions': _stringify_keys(trial.distributions)})

    def _log_best_trials(self, study):

        self.run.log(_log_best_trials(study))

    def _log_study_details(self, study, trial):
        if trial._trial_id == 0:
            _log_study_details(self.run, study)

    def _log_plots(self, study, trial):
        if self._should_log_plots(study, trial):
            _log_plots(self.run, study,
                       visualization_backend=self._visualization_backend,
                       log_plot_contour=self._log_plot_contour,
                       log_plot_edf=self._log_plot_edf,
                       log_plot_parallel_coordinate=self._log_plot_parallel_coordinate,
                       log_plot_param_importances=self._log_plot_param_importances,
                       log_plot_pareto_front=self._log_plot_pareto_front,
                       log_plot_slice=self._log_plot_slice,
                       log_plot_optimization_history=self._log_plot_optimization_history,
                       log_plot_intermediate_values=self._log_plot_intermediate_values,
                       )

    def _log_study(self, study, trial):
        if self._should_log_study(trial):
            _log_study(self.run, study)

    def _should_log_plots(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        if not len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))):
            return False
        elif self._plots_update_freq == 'never':
            return False
        else:
            if trial._trial_id % self._plots_update_freq == 0:
                return True
        return False

    def _should_log_study(self, trial: optuna.trial.FrozenTrial):
        if self._study_update_freq == 'never':
            return False
        else:
            if trial._trial_id % self._study_update_freq == 0:
                return True
        return False


def log_study_metadata(study: optuna.Study,
                       run,
                       log_plots=True,
                       log_study=True,
                       log_all_trials=True,
                       log_distributions=True,
                       visualization_backend='plotly',
                       log_plot_contour=True,
                       log_plot_edf=True,
                       log_plot_parallel_coordinate=True,
                       log_plot_param_importances=True,
                       log_plot_pareto_front=True,
                       log_plot_slice=True,
                       log_plot_intermediate_values=True,
                       log_plot_optimization_history=True):
    """A function that logs the metadata from Optuna Study to Neptune.

    With this function, you can log and display:

    * values and params for each trial
    * current best values and params for the study
    * visualizations from the `optuna.visualizations` module
    * parameter distributions for each trial
    * study object itself to load it later
    * and more

    Args:
        study(optuna.Study): Optuna study object.
        run(neptune.Run): Neptune Run.
        base_namespace(str, optional): Namespace inside the Run where your study metadata is logged. Defaults to ''.
        log_plots(bool): If 'True' the visualiztions from `optuna.visualizations` will be logged to Neptune.
            Defaults to 'True'.
        log_study(bool): If 'True' the study will be logged to Neptune. Depending on the study storage type used
            different objects are logged. If 'InMemoryStorage' is used the pickled study
            object will be logged to Neptune. Otherwise database URL will be logged. Defaults to 'True'.
        log_all_trials(bool): If 'True' all trials are logged. Defaults to 'True'.
        log_distributions(bool): If 'True' the distributions for all trials are logged. Defaults to 'True'.
        visualization_backend(str, optional): Which visualization backend is used for 'optuna.visualizations' plots.
            It can be one of 'matplotlib' or 'plotly'. Defaults to 'plotly'.
        log_plot_contour(bool, optional): If 'True' the `optuna.visualizations.plot_contour`
            visualization will be logged to Neptune. Defaults to `True`.
        log_plot_edf(bool, optional): If 'True' the `optuna.visualizations.plot_edf`
            visualization will be logged to Neptune. Defaults to `True`.
        log_plot_parallel_coordinate(bool, optional): If 'True' the `optuna.visualizations.plot_parallel_coordinate`
            visualization will be logged to Neptune. Defaults to `True`.
        log_plot_param_importances(bool, optional): If 'True' the `optuna.visualizations.plot_param_importances`
            visualization will be logged to Neptune. Defaults to `True`.
        log_plot_pareto_front(bool, optional): If 'True' the `optuna.visualizations.plot_pareto_front`
            visualization will be logged to Neptune.
            If your `optuna.study` is not multi-objective this plot is not logged. Defaults to `True`.
        log_plot_slice(bool, optional): If 'True' the `optuna.visualizations.plot_slice`
            visualization will be logged to Neptune. Defaults to `True`.
        log_plot_intermediate_values(bool, optional): If 'True' the `optuna.visualizations.plot_intermediate_values`
            visualization will be logged to Neptune.
            If your `optuna.study` is not using pruners this plot is not logged. Defaults to `True`. Defaults to `True`.
        log_plot_optimization_history(bool, optional): If 'True' the `optuna.visualizations.plot_optimization_history`
            visualization will be logged to Neptune. Defaults to `True`.
    """
    _log_study_details(run, study)
    log_dict = _log_best_trials(study)

    if log_all_trials:
        _log_trials(run, study.trials)

    if log_distributions:
        log_dict['study/distributions'] = list(trial.distributions for trial in study.trials)
    run.log(log_dict)

    if log_plots:
        _log_plots(run, study,
                   visualization_backend=visualization_backend,
                   log_plot_contour=log_plot_contour,
                   log_plot_edf=log_plot_edf,
                   log_plot_parallel_coordinate=log_plot_parallel_coordinate,
                   log_plot_param_importances=log_plot_param_importances,
                   log_plot_pareto_front=log_plot_pareto_front,
                   log_plot_slice=log_plot_slice,
                   log_plot_optimization_history=log_plot_optimization_history,
                   log_plot_intermediate_values=log_plot_intermediate_values,
                   )

    if log_study:
        _log_study(run, study)


def _log_study_details(run, study: optuna.Study):
    log_dict = dict()
    log_dict['study/study_name'] = study.study_name
    log_dict['study/direction'] = _stringify_keys(study.direction)
    log_dict['study/directions'] = _stringify_keys(study.directions)
    log_dict['study/system_attrs'] = study.system_attrs
    log_dict['study/user_attrs'] = study.user_attrs
    try:
        log_dict['study/_study_id'] = study._study_id
        log_dict['study/_storage'] = _stringify_keys(study._storage)
    except AttributeError:
        pass
    run.log(log_dict)


def _log_study(run, study: optuna.Study):
    try:
        log_dict = dict()
        if type(study._storage) is optuna.storages._in_memory.InMemoryStorage:
            """pickle and log the study object to the 'study/study.pkl' path"""
            log_dict['study/study_name'] = study.study_name
            log_dict['study/storage_type'] = 'InMemoryStorage'
            joblib.dump(study, f"{wandb.run.dir}/optuna_study.pkl")  # save study
            run.save(f"{wandb.run.dir}/optuna_study.pkl")
            pass
        else:
            log_dict['study/study_name'] = study.study_name
            if isinstance(study._storage, optuna.storages.RedisStorage):
                log_dict['study/storage_type'] = "RedisStorage"
                log_dict['study/storage_url'] = study._storage._url
            elif isinstance(study._storage, optuna.storages._CachedStorage):
                log_dict['study/storage_type'] = "RDBStorage"  # apparently CachedStorage typically wraps RDBStorage
                log_dict['study/storage_url'] = study._storage._backend.url
            elif isinstance(study._storage, optuna.storages.RDBStorage):
                log_dict['study/storage_type'] = "RDBStorage"
                log_dict['study/storage_url'] = study._storage.url
            else:
                log_dict['study/storage_type'] = "unknown storage type"
                log_dict['study/storage_url'] = "unknown storage url"
        run.log(log_dict)
    except AttributeError:
        return


def _log_plots(run,
               study: optuna.Study,
               visualization_backend='plotly',
               log_plot_contour=True,
               log_plot_edf=True,
               log_plot_parallel_coordinate=True,
               log_plot_param_importances=True,
               log_plot_pareto_front=True,
               log_plot_slice=True,
               log_plot_intermediate_values=True,
               log_plot_optimization_history=True,
               ):
    if visualization_backend == 'matplotlib':
        import optuna.visualization.matplotlib as vis
    elif visualization_backend == 'plotly':
        import optuna.visualization as vis
    else:
        raise NotImplementedError(f'{visualization_backend} visualisation backend is not implemented')

    log_img_dict = dict()
    if vis.is_available:
        params = list(p_name for t in study.trials for p_name in t.params.keys())

        if log_plot_contour and any(params):
            log_img_dict['visualizations/plot_contour'] = vis.plot_contour(study)

        if log_plot_edf:
            log_img_dict['visualizations/plot_edf'] = vis.plot_edf(study)

        if log_plot_parallel_coordinate:
            log_img_dict['visualizations/plot_parallel_coordinate'] = vis.plot_parallel_coordinate(study)

        if log_plot_param_importances and len(
                study.get_trials(states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED,))) > 1:
            log_img_dict['visualizations/plot_param_importances'] = vis.plot_param_importances(study)

        if log_plot_pareto_front and study._is_multi_objective() and visualization_backend == 'plotly':
            log_img_dict['visualizations/plot_pareto_front'] = vis.plot_pareto_front(study)

        if log_plot_slice and any(params):
            log_img_dict['visualizations/plot_slice'] = vis.plot_slice(study)

        if log_plot_intermediate_values and any(trial.intermediate_values for trial in study.trials):
            # Intermediate values plot if available only if the above condition is met
            log_img_dict['visualizations/plot_intermediate_values'] = vis.plot_intermediate_values(study)

        if log_plot_optimization_history:
            log_img_dict['visualizations/plot_optimization_history'] = vis.plot_optimization_history(study)
        run.log(log_img_dict)


def _log_best_trials(study: optuna.Study):
    if not study.best_trials:
        return dict()

    best_results = {'value': study.best_value,
                    'params': study.best_params,
                    'value|params': f'value: {study.best_value}| params: {study.best_params}'}

    prefix = 'study/best/trials'
    best_results = {f'{prefix}/k': v for k, v in best_results.items()}
    for trial in study.best_trials:
        best_results[f'{prefix}/{trial._trial_id}/datetime_start'] = str(trial.datetime_start)
        best_results[f'{prefix}/{trial._trial_id}/datetime_complete'] = str(trial.datetime_complete)
        best_results[f'{prefix}/{trial._trial_id}/duration'] = str(trial.duration)
        best_results[f'{prefix}/{trial._trial_id}/distributions'] = str(trial.distributions)
        best_results[f'{prefix}/{trial._trial_id}/intermediate_values'] = trial.intermediate_values
        best_results[f'{prefix}/{trial._trial_id}/params'] = trial.params
        best_results[f'{prefix}/{trial._trial_id}/value'] = trial.value
        best_results[f'{prefix}/{trial._trial_id}/values'] = trial.values

    return best_results


def _log_trials(run, trials: Iterable[optuna.trial.FrozenTrial]):
    log_dict = dict()
    for trial in trials:
        # if trial.value:
        #    log_dict['values'].log(trial.value, step=trial._trial_id)

        # log_dict['params'].log(trial.params)
        log_dict[f'trials/{trial._trial_id}/datetime_start'] = trial.datetime_start
        # log_dict[f'trials/{trial._trial_id}/datetime_complete'] = trial.datetime_complete
        # log_dict[f'trials/{trial._trial_id}/duration'] = trial.duration
        log_dict[f'trials/{trial._trial_id}/distributions'] = _stringify_keys(trial.distributions)
        # log_dict[f'trials/{trial._trial_id}/intermediate_values'] = _stringify_keys(trial.intermediate_values)
        log_dict[f'trials/{trial._trial_id}/params'] = _stringify_keys(trial.params)
        # log_dict[f'trials/{trial._trial_id}/value'] = trial.value
        # log_dict[f'trials/{trial._trial_id}/values'] = trial.values
    run.log(log_dict)


def _stringify_keys(o):
    return {str(k): _stringify_keys(v) for k, v in o.items()} if isinstance(o, dict) else str(o)


def _get_pickle(run, filename):
    with tempfile.TemporaryDirectory() as d:
        run.restore(filename)
        filepath = os.listdir(d)[0]
        full_path = os.path.join(d, filepath)
        artifact = joblib.load(full_path)

    return artifact


def reload_study_from_wandb(run_path):
    with tempfile.TemporaryDirectory() as d:
        wandb.restore('optuna_study.pkl', run_path=run_path, replace=True, root=d)
        filepath = os.listdir(d)[0]
        full_path = os.path.join(d, filepath)
        study = joblib.load(full_path)
    return study
# ----------------------------------------------------------------------------------------------------------------------

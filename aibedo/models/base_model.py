import logging
import time
from typing import Optional, List, Any, Dict, Sequence, Union

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import torchmetrics
import importlib

if importlib.util.find_spec("wandb"):
    import wandb

from aibedo.data_transforms.normalization import get_variable_stats, get_clim_err, destandardize, standardize, \
    rescale_nonorm_vars
from aibedo.utilities.samplings import icosahedron_nodes_calculator
from aibedo.utilities.constraints import nonnegative_precipitation, global_moisture_constraint, \
    mass_conservation_constraint, AUXILIARY_VARS, precipitation_energy_budget_constraint
from aibedo.utilities.utils import get_logger, to_DictConfig, get_loss, raise_error_if_invalid_value, \
    stem_var_id, get_input_var_to_idx


class BaseModel(LightningModule):
    r""" This is a template base class, that should be inherited by any AIBEDO stand-alone ML model.
    Methods that need to be implemented by your concrete ML model (just as if you would define a :class:`torch.nn.Module`):
        - :func:`__init__`
        - :func:`forward`

    The other methods may be overridden as needed.
    It is recommended to define the attribute
        >>> self.example_input_array = torch.randn(<YourModelInputShape>)  # batch dimension can be anything, e.g. 7


    .. note::
        Please use the function :func:`predict` at inference time for a given input tensor, as it postprocesses the
        raw predictions from the function :func:`raw_predict` (or model.forward or model())!

    Args:
        datamodule_config: DictConfig with the configuration of the datamodule
        optimizer: DictConfig with the optimizer configuration (e.g. for AdamW)
        scheduler: DictConfig with the scheduler configuration (e.g. for CosineAnnealingLR)
        monitor (str): The name of the metric to monitor, e.g. 'val/mse'
        mode (str): The mode of the monitor. Default: 'min' (lower is better)
        window (int): How many time-steps to use for prediction. Default: 1
        loss_weights: The weights for each of the sub-losses for each output variable. Default: Uniform weights
        physics_loss_weights: The weights for each of the physics losses. Default: No physics loss (all zeros)
        input_noise_std (float): The standard deviation of the Gaussian noise to add to the input data. Default: 0
        nonnegativity_at_train_time (bool): Whether to enforce non-negativity at train time/ for backprop. Only used if physics_loss_weights[3] > 0
        month_as_feature (bool or str): Whether/How to use the month as a feature. Default: ``False`` (i.e. do not use it)
        use_auxiliary_vars (bool): Whether to use the auxiliary variables for computing the
            physics constraint losses (regardless of whether they are penalized). Default: ``True``
        loss_function (str): The name of the loss function. Default: 'mean_squared_error'
        name (str): optional string with a name for the model
        verbose (bool): Whether to print/log or not

    Read the docs regarding LightningModule for more information:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(self,
                 datamodule_config: DictConfig = None,
                 optimizer: Optional[DictConfig] = None,
                 scheduler: Optional[DictConfig] = None,
                 monitor: Optional[str] = None,
                 mode: str = "min",
                 window: int = 1,
                 loss_weights: Union[Sequence[float], Dict[str, float]] = (0.33, 0.33, 0.33),
                 physics_loss_weights: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0),
                 lambda_physics1: float = None,
                 lambda_physics2: float = None,
                 lambda_physics3: float = None,
                 lambda_physics4: bool = None,
                 lambda_physics5: float = None,
                 input_noise_std: float = 0.0,
                 nonnegativity_at_train_time: bool = True,
                 month_as_feature: Union[bool, str] = False,
                 use_auxiliary_vars: bool = True,
                 loss_function: str = "mean_squared_error",
                 name: str = "",
                 verbose: bool = True,
                 input_transform=None,  # todo: deprecate argument
                 ):
        super().__init__()
        # The following saves all the args that are passed to the constructor to self.hparams
        #   e.g. access them with self.hparams.monitor
        self.save_hyperparameters()
        # Get a logger
        self.log_text = get_logger(name=self.__class__.__name__ if name == '' else name)
        self.name = name
        self.verbose = verbose
        if not self.verbose:  # turn off info level logging
            self.log_text.setLevel(logging.WARN)

        self.AUX_VARS = AUXILIARY_VARS if use_auxiliary_vars else []
        self._input_var_to_idx, _ = get_input_var_to_idx(
            datamodule_config.input_vars, self.AUX_VARS, window=self.hparams.window
        )
        self.main_input_vars = datamodule_config.input_vars

        # Infer the data dimensions
        self._data_dir = datamodule_config.data_dir
        if 'order' in datamodule_config:
            self.spatial_dim = n_pixels = icosahedron_nodes_calculator(datamodule_config.order)
            ico = "isosph5" if datamodule_config.order == 5 else "isosph"
            self.data_files_id = f'compress.{ico}'
        else:
            self.spatial_dim = (192, 288)
            self.data_files_id = ''
        self._num_input_features = in_channels = len(datamodule_config.input_vars)
        self._num_output_features = out_channels = len(datamodule_config.output_vars)
        self._output_vars = datamodule_config.output_vars
        for var in self._output_vars:
            var_id = stem_var_id(var)
            setattr(self, f'_{var_id}_name', var)
        # edit actual #features when using month information
        if self.hparams.month_as_feature == 'one_hot':
            self._num_input_features += 12
        elif self.hparams.month_as_feature:
            self._num_input_features += 1

        # loss function (one per target variable)
        self.loss_weights = loss_weights
        self.criterion = {v: get_loss(loss_function, reduction='mean') for v in self.output_vars}

        # Timing variables to track the training/epoch/validation time
        self._start_validation_epoch_time = self._start_test_epoch_time = self._start_epoch_time = None

        # Metrics
        val_metrics = {'val/mse': torchmetrics.MeanSquaredError(squared=True)}
        for output_var in self.output_vars:
            var_id = stem_var_id(output_var)
            val_metrics[f'val/{output_var}/rmse'] = torchmetrics.MeanSquaredError(squared=False)
            val_metrics[f'val/{var_id}/rmse'] = torchmetrics.MeanSquaredError(squared=False)

        self.val_metrics = nn.ModuleDict(val_metrics)
        self._test_metrics = self._predict_metrics = None
        # Check that the args/hparams are valid
        self._check_args()

        # Set the target/auxiliary variable statistics needed
        stats_kwargs = dict(data_dir=self.data_dir, files_id=self.data_files_id)
        # Go through all output (and auxiliary) vars:
        vars_with_stats = self.output_vars + self.AUX_VARS if use_auxiliary_vars else self.output_vars
        for output_var in vars_with_stats:
            var_id = stem_var_id(output_var)
            var_mean, var_std = get_variable_stats(var_id=var_id, **stats_kwargs)
            self.register_buffer_dummy(f'{var_id}_mean', var_mean, persistent=False)
            self.register_buffer_dummy(f'{var_id}_std', var_std, persistent=False)

        if use_auxiliary_vars:
            # Go through all climatology auxiliary arrays:
            for aux_clim_err in ['Precip', 'PE', 'PS']:
                err_id = f'{aux_clim_err}_clim_err'
                err_arr = get_clim_err(err_id=err_id, **stats_kwargs)
                self.register_buffer_dummy(err_id, err_arr, persistent=False)

        if physics_loss_weights[1] > 0:
            self.log_text.info(" Using constraint (#2)")
        if physics_loss_weights[2] > 0:
            self.log_text.info(" Using global moisture constraint (#3)")
        if physics_loss_weights[3] > 0:
            self.log_text.info(" Using non-negative precipitation constraint (#4)")
        if physics_loss_weights[4] > 0:
            self.log_text.info(" Using mass conservation constraint (#5)")

    @property
    def num_input_features(self) -> int:
        return self._num_input_features * self.hparams.window

    @property
    def num_output_features(self) -> int:
        return self._num_output_features

    @property
    def input_var_to_idx(self) -> Dict[str, int]:
        """ Returns the index of the month (the month being a scalar in {0, 1, .., 11}) in the input data """
        # if self._input_var_to_idx is None:
        # if self._trainer is not None:
        #    self._input_var_to_idx = self.trainer.datamodule.input_var_to_idx
        return self._input_var_to_idx

    @property
    def test_set_name(self) -> str:
        return self.trainer.datamodule.test_set_name if hasattr(self.trainer.datamodule, 'test_set_name') else 'test'

    @property
    def prediction_set_name(self) -> str:
        return self.trainer.datamodule.prediction_set_name if hasattr(self.trainer.datamodule,
                                                                      'prediction_set_name') else 'predict'

    @property
    def test_metrics(self):
        if self._test_metrics is None:
            metrics = {f'{self.test_set_name}/mse': torchmetrics.MeanSquaredError(squared=True)}
            for output_var in self.output_vars:
                var_id = stem_var_id(output_var)
                metrics[f'{self.test_set_name}/{output_var}/rmse'] = torchmetrics.MeanSquaredError(squared=False)
                metrics[f'{self.test_set_name}/{var_id}/rmse'] = torchmetrics.MeanSquaredError(squared=False)
                metrics[f'{self.test_set_name}/{var_id}/mae'] = torchmetrics.MeanAbsoluteError()

            self._test_metrics = nn.ModuleDict(metrics).to(self.device)
        return self._test_metrics

    @property
    def predict_metrics(self):
        if self._predict_metrics is None:
            metrics = dict()
            for output_var in self.output_vars:
                var_id = stem_var_id(output_var)
                metrics[f'{self.prediction_set_name}/{var_id}/rmse'] = torchmetrics.MeanSquaredError(squared=False)
                metrics[f'{self.prediction_set_name}/{var_id}/mae'] = torchmetrics.MeanAbsoluteError()

            self._predict_metrics = nn.ModuleDict(metrics).to(self.device)
        return self._predict_metrics

    @property
    def n_params(self):
        """ Returns the number of parameters in the model """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def data_dir(self) -> str:
        if self._data_dir is None:
            self._data_dir = self.trainer.datamodule.hparams.data_dir
        return self._data_dir

    @property
    def output_vars(self) -> List[str]:
        if self._output_vars is None:
            self._output_vars = self.trainer.datamodule.hparams.output_vars
        return self._output_vars

    @property
    def logging_output_var_names(self) -> List[str]:
        return self.output_vars + [stem_var_id(x) for x in self.output_vars]

    def _check_args(self):
        """Check if the arguments are valid."""
        assert self.hparams.window > 0, "Window size must be greater than 0"
        plw = self.hparams.physics_loss_weights
        if len(plw) != 5:
            raise ValueError(f'The number of physics loss weights must be 5, but got {plw}')
        if plw[3] > 0 and plw[3] != 1:
            self.log_text.info(f'The fourth physics loss weight must be 0 or 1, but got {plw[3]}. Setting it to 1.')
            self.hparams.physics_loss_weights[3] = 1
        lw = self.loss_weights
        if isinstance(lw, dict) and len(lw.keys()) != len(self.output_vars) or \
                isinstance(lw, Sequence) and len(lw) != len(self.output_vars):
            raise ValueError(f'The number of loss weights must be same as #output-vars={len(self.output_vars)}'
                             f', but got {lw}')
        if isinstance(lw, Sequence) or isinstance(lw, list) or isinstance(lw, tuple):
            self.loss_weights = {v: lw[i] for i, v in enumerate(self.output_vars)}
        if not any(w > 0 for w in self.loss_weights.values()):
            raise ValueError(f'At least one loss weight must be > 0, but got {self.loss_weights}')

        raise_error_if_invalid_value(self.hparams.month_as_feature, [False, True, 'one_hot'], 'month_as_feature')

    def forward(self, X: Tensor):
        r""" Standard ML model forward pass (to be implemented by the specific ML model).

        Args:
            X (Tensor): Input data tensor of shape :math:`(B, *, C_{in})`
        Shapes:
            - Input: :math:`(B, *, C_{in})`,

            where :math:`B` is the batch size, :math:`*` is the spatial dimension(s) of the data,
            and :math:`C_{in}` is the number of input features/channels.
        """
        raise NotImplementedError('Base model is an abstract class!')

    def _raw_predict_single_tensor(self, X: Tensor) -> Tensor:
        r"""
        Predict the raw (normalized) output of the model.
        To get the predictions with post-processing
        (e.g. non-negativity, denormalization, split predictions by output variable),
        please use :func:`predict` instead.
        To only get the predictions by output variable, please use :func:`raw_predict` instead.

        Args:
            X (Tensor): Input data tensor of shape :math:`(B, *, C_{in})`, where * refers to the spatial dimension(s) of the data.
                This is the same tensor one would use in :func:`forward`,
                except for (potentially) auxiliary inputs (e.g. the month of the outputs) that will be transformed or
                removed before passing to the model through :func:`forward`.

        Returns:
            Tensor: The model's prediction tensor of shape :math:`(B, *, C_{out})`.
                Note that these predictions are in normalized scale.

        Shapes:
            - Input: :math:`(B, *, C_{in})`
            - Output: :math:`(B, *, C_{out})`,

            where :math:`B` is the batch size, :math:`*` is the spatial dimension(s) of the data,
            and :math:`C_{in}` (:math:`C_{out}`) is the number of input (output) features.
        """
        if self.hparams.month_as_feature == 'one_hot':
            # Add one-hot encoding of the month
            MONTH_IDX = self.input_var_to_idx['month']
            X_feats = X[..., :MONTH_IDX]  # raw inputs have raw month encoding (single scalar)!
            X_mon = X[..., MONTH_IDX].long()  # month is a scalar
            X_mon_one_hot = F.one_hot(X_mon, num_classes=12)
            X_feats = torch.cat([X_feats, X_mon_one_hot], dim=-1)
        else:
            # Just remove the potentially added auxiliary vars
            X_feats = X[..., :self.num_input_features]

        # Do the following only during training
        if self.training:
            if self.hparams.input_noise_std > 0:
                # Add white noise to the input
                X_feats = X_feats + torch.randn_like(X_feats) * self.hparams.input_noise_std
            # Add red noise to the input
            # X_feats = X_feats + torch.randn_like(X_feats) * self.hparams.input_noise_std * torch.arange(X_feats.shape[-1], device=X_feats.device)
        Y = self(X_feats)  # forward pass
        return Y

    def _split_raw_preds_per_target_variable(self, outputs_tensor: Tensor) -> Dict[str, Tensor]:
        r"""
        Split the output/predicted/target tensor into a dictionary of tensors with a key for each output variable in the model.

        Args:
            outputs_tensor: A tensor of shape :math:`(B, *, C_{out})`

        Returns:
            Dict[str, Tensor]: A dictionary of tensors of shape :math:`(B, *)` for each output variable in the model.
                E.g. 'pr_pre', 'tas_pre' will all be the keys to the respective predicted/target tensor.
        """
        if outputs_tensor.shape[-1] != len(self.output_vars):
            raise ValueError(
                f"outputs_tensor.shape[-1] = {outputs_tensor.shape[-1]},"
                f" but #output-vars = {len(self.output_vars)}")
        preds_per_target_variable = {
            var_name: outputs_tensor[..., i]  # index the tensor along the last dimension
            for i, var_name in enumerate(self.output_vars)
        }
        return preds_per_target_variable

    def raw_predict(self, X: Tensor) -> Dict[str, Tensor]:
        r"""
        Predict the raw (normalized) output of the model, splitted into a dict by output variable.
        To get the predictions with post-processing
        (e.g. non-negativity, denormalization),
        please use :func:`predict` instead.

        Args:
            X (Tensor): Input data tensor of shape :math:`(B, *, C_{in})`, where * refers to the spatial dimension(s) of the data.
                This is the same tensor one would use in :func:`forward`,
                except for (potentially) auxiliary inputs (e.g. the month of the outputs) that will be transformed or
                removed before passing to the model through :func:`forward`.

        Returns:
            Dict[str, Tensor]: A dictionary with :math:`C_{out}` entries of tensors of shape :math:`(B, *)`.
                E.g. 'pr_pre', 'tas_pre' (or 'pre_nonorm', 'tas_nonorm') will all be the keys to the respective predicted tensor.
                Note that these predictions are in normalized scale.

        Shapes:
            - Input: :math:`(B, *, C_{in})`
            - Output: Dict k_i -> v_i, and each v_i has shape :math:`(B, *)` for :math:`i=1,..,C_{out}`,

            where :math:`B` is the batch size, :math:`*` is the spatial dimension(s) of the data,
            and :math:`C_{out}` is the number of output features,
        """
        outputs_tensor = self._raw_predict_single_tensor(X)
        preds_per_target_variable = self._split_raw_preds_per_target_variable(outputs_tensor)
        return preds_per_target_variable

    def _get_monthly_data_for_batch(self, month_to_data: Tensor, months_of_batch: Tensor, dim: int = 0, **kwargs):
        """

        Args:
            month_to_data: A tensor of shape (12, *) if dim=0, or (*, 12, *) if dim=1, etc.
            months_of_batch: The months of the batch, of shape (B, ). Must be a LongTensor!
            dim: The dimension to index the month_to_data tensor along
            **kwargs: Additional arguments to pass to torch.index_select

        Returns:
            A tensor of shape (B, *)
        """
        assert month_to_data.shape[
                   dim] == 12, f'The dimension {dim} of month_to_data must be 12, but got {month_to_data.shape[dim]}'
        # torch.index_select(.) will ensure that the correct monthly mean/std is used for each example/snapshot:
        #  E.g. pr_mean has shape (12, num_grid_cells), and the use of index_select assumes that index 0 is for january,
        #  index 1 for february, etc. (i.e. index of first dimension is the month)
        #  It will then return the associated spatial mean/std for each example/snapshot's month
        return torch.index_select(month_to_data, index=months_of_batch, dim=dim, **kwargs)

    def _denormalize_variable(self,
                              normalized_var: Tensor,
                              month_of_var: Tensor,
                              output_var_name: str,
                              return_monthly_stats: bool = False
                              ) -> Tensor:
        """
        Denormalize the normalized variable, i.e. bring it back into its original scale (e.g. Kelvin for temperature)
        Args:
            normalized_var: The variable in normalized scale, of shape (B, *)
            month_of_var: The month of each batch element, of shape (B, ). Must be a LongTensor!
            output_var_name: The name of the variable (e.g. 'tas'). It does not matter if suffixed with '_pre' or not.
            return_monthly_stats: Whether to return the monthly mean and std of the variable too.

        Returns:

        """
        var_id = stem_var_id(output_var_name)
        if hasattr(self, f'{var_id}_mean'):
            mean, std = getattr(self, f'{var_id}_mean'), getattr(self, f'{var_id}_std')
        else:
            # try to get the mean and std from the data directory
            mean, std = get_variable_stats(var_id=var_id, data_dir=self.data_dir, files_id=self.data_files_id)
        mean, std = mean.type_as(normalized_var), std.type_as(normalized_var)

        batch_monthly_mean = self._get_monthly_data_for_batch(mean, months_of_batch=month_of_var, dim=0)
        batch_monthly_std = self._get_monthly_data_for_batch(std, months_of_batch=month_of_var, dim=0)

        if '_nonorm' in output_var_name:
            # Rescale _nonorm outputs to the original units
            normalized_var = rescale_nonorm_vars({output_var_name: normalized_var})[output_var_name]
            denormed_var = normalized_var + batch_monthly_mean
        elif '_pre' in output_var_name:
            denormed_var = destandardize(normalized_var, batch_monthly_mean, batch_monthly_std)
        else:
            raise ValueError(f"output_var_name must be suffixed with '_pre' or '_nonorm'")
        if return_monthly_stats:
            return denormed_var, batch_monthly_mean, batch_monthly_std
        return denormed_var

    def raw_outputs_to_denormalized_per_variable_dict(self,
                                                      outputs_tensor: Tensor,
                                                      month_of_outputs: Tensor = None,
                                                      input_tensor: Tensor = None,
                                                      return_normalized_outputs: bool = False
                                                      ) -> Dict[str, Tensor]:
        """
        Convert the output tensor into a dictionary of denormalized (!) per-output-variable tensors.
        This function can be used for target tensors. For predicted tensors, please use :func:`postprocess_raw_predictions`.

        Args:
            outputs_tensor (Tensor): A tensor of shape :math:`(B, *, C_{out})` in normalized scale.
            month_of_outputs (Tensor, optional): A tensor of shape :math:`(B,)` with the month of each output.
            input_tensor (Tensor, optional): A tensor of shape :math:`(B, *, C_{in})`.
            return_normalized_outputs (bool): If True, the raw outputs (vars with '_pre') will be returned as well.

        .. note::

            One of ``month_of_outputs`` or ``input_tensor`` must be provided!

        Returns:
            Dict[str, Tensor]: A dictionary of denormalized tensors of shape :math:`(B, *)`.
                E.g. 'pr', 'tas' will all be the keys to the respective predicted/target tensor.

        Shapes:
            - Input: :math:`(B, *, C_{out})`
            - Output: Dict k_i -> v_i, and each v_i has shape :math:`(B, *)` for :math:`i=1,..,C_{out}`,

            where :math:`B` is the batch size, :math:`*` is the spatial dimension(s) of the data,
            and :math:`C_{out}` is the number of output features,
        """
        assert month_of_outputs is not None or outputs_tensor is not None, "Either month_of_outputs or outputs_tensor must be provided!"
        # Get the split {var: out_var_tensor} dictionary, with var, e.g., \in {'pr', 'tas', ...}
        preds_per_target_variable = self._split_raw_preds_per_target_variable(outputs_tensor)
        # Retrieve the months of the batch if not provided
        if month_of_outputs is None:
            MONTH_IDX = self.input_var_to_idx['month']  # index (in the channel dim) of the month in the input tensor
            # idx 0 is arbitrary
            if len(input_tensor.shape) == 4:
                month_of_outputs = input_tensor[:, 0, 0, MONTH_IDX]
            else:
                month_of_outputs = input_tensor[:, 0, MONTH_IDX]
            # month_of_outputs has shape (batch_size,)
        # Convert the months of the batch tensor to a LongTensor (needed for index_select indexing with it)
        month_of_outputs = month_of_outputs.type_as(outputs_tensor).long()
        # Denormalize the outputs (using the per-variable means and standard deviations)
        denormed_Y_per_target_variable = dict()
        for var_name, var_tensor in preds_per_target_variable.items():
            var_id = stem_var_id(var_name)  # use, e.g. 'pr' instead of 'pr_pre' for the original-scale variable
            denormed_Y_per_target_variable[var_id] = self._denormalize_variable(var_tensor, month_of_outputs, var_name)
        if return_normalized_outputs:
            # Return the raw outputs (vars with '_pre') as well
            return {**denormed_Y_per_target_variable, **preds_per_target_variable}
        # Only return the denormalized outputs (vars without '_pre')
        return denormed_Y_per_target_variable

    def postprocess_raw_predictions(self, preds_tensor: Tensor, **kwargs) -> Dict[str, Tensor]:
        r"""Convert the raw model predictions to post-processed predictions.
        Post-processing includes:
            - denormalization  (bring the predictions to the original scale)
            - enforcing non-negative values  (e.g. for precipitation)
            - Splitting the predictions per target variable into a dictionary of output_var -> output_var_prediction.

        Args:
            preds_tensor (Tensor): Raw/unprocessed predictions of shape :math:`(B, *, C_{out})`.
            **kwargs: Additional keyword arguments to be passed to :func:`postprocess_raw_predictions`

        Returns:
            Dict[str, Tensor]: The model predictions (in a post-processed format), i.e. a dictionary output_var -> output_var_prediction,
                where each output_var_prediction is a Tensor of shape :math:`(B, *)` in original-scale (e.g.
                in Kelvin for temperature), and non-negativity has been enforced for variables such as precipitation.

        Shapes:
            - Input: :math:`(B, *, C_{out})`
            - Output: Dict :math:`k_i` -> :math:`v_i`, and each :math:`v_i` has shape :math:`(B, *)` for :math:`i=1,..,C_{out}`,

            where :math:`B` is the batch size, :math:`*` is the spatial dimension(s) of the data,
            and :math:`C_{out}` is the number of output features.
        """
        preds_dict: Dict[str, Tensor] = self.raw_outputs_to_denormalized_per_variable_dict(preds_tensor, **kwargs)
        # Enforce non-negative output variables (constraint 4)
        if self.hparams.physics_loss_weights[3] > 0:
            ps_or_psl = 'psl' if 'psl' in preds_dict.keys() else 'ps'
            preds_dict['pr'] = nonnegative_precipitation(preds_dict['pr'])
            preds_dict[ps_or_psl] = nonnegative_precipitation(preds_dict[ps_or_psl])
            preds_dict['tas'] = nonnegative_precipitation(preds_dict['tas'])

        return preds_dict

    def predict(self, X: Tensor, **kwargs) -> Dict[str, Tensor]:
        """
        This should be the main method to use for making predictions.

        Args:
            X (Tensor): Input data tensor of shape :math:`(B, *, C_{in})`.
                This is the same tensor one would use in :func:`forward`,
                except for (potentially) auxiliary inputs (e.g. the month of the outputs) that will be transformed or
                removed before passing to the model through :func:`forward`.
            **kwargs: Additional keyword arguments to be passed to :func:`postprocess_raw_predictions`

        Returns:
            Dict[str, Tensor]: The model predictions (in a post-processed format), i.e. a dictionary output_var -> output_var_prediction,
                where each output_var_prediction is a Tensor of shape :math:`(B, *)` in original-scale (e.g.
                in Kelvin for temperature), and non-negativity has been enforced for variables such as precipitation.

        Shapes:
            - Input: :math:`(B, *, C_{in})`
            - Output: Dict :math:`k_i` -> :math:`v_i`, and each :math:`v_i` has shape :math:`(B, *)` for :math:`i=1,..,C_{out}`,

            where :math:`B` is the batch size, :math:`*` is the spatial dimension(s) of the data,
            and :math:`C_{out}` is the number of output features.
        """
        raw_preds: Tensor = self._raw_predict_single_tensor(X)
        preds: Dict[str, Tensor] = self.postprocess_raw_predictions(raw_preds, input_tensor=X, **kwargs)
        return preds

    # --------------------- training with PyTorch Lightning
    def on_train_start(self) -> None:
        """ Log some info about the model/data at the start of training """
        if not self.hparams.use_auxiliary_vars:
            psw = self.hparams.physics_loss_weights
            if psw[0] > 0 or psw[1] > 0 or psw[2] > 0 or psw[4] > 0:
                raise ValueError(
                    "The model is configured to not use auxiliary variables, but the physics_loss_weights are > 0!")
        self.log('Parameter count', float(self.n_params))
        self.log('Training set size', float(len(self.trainer.datamodule._data_train)))
        self.log('Validation set size', float(len(self.trainer.datamodule._data_val)))

    def on_train_epoch_start(self) -> None:
        self._start_epoch_time = time.time()

    def train_step_initial_log_dict(self) -> dict:
        return dict()

    def training_step(self, batch: Any, batch_idx: int):
        """ One step of training (Backpropagation is done on the loss returned at the end of this function) """
        X, Y = batch
        train_log = self.train_step_initial_log_dict()

        # Predict normalized outputs and split them into output_var: Tensor preds/targets of that var
        preds = self.raw_predict(X)
        Y = self._split_raw_preds_per_target_variable(Y)

        # Get the month of each example (the month is the same for all grid cells in an example)
        # idx 0 is arbitrary
        if len(X.shape) == 4:  # 2D spatial data
            month_of_batch = X[:, 0, 0, self.input_var_to_idx['month']].long()  # has shape (batch_size,)
        else:  # Spherical data
            month_of_batch = X[:, 0, self.input_var_to_idx['month']].long()  # has shape (batch_size,)

        if self.hparams.physics_loss_weights[3] > 0:
            # Enforce non-negative precipitation (constraint 4); needs to be done before the main loss is computed
            tmp_pr_normed, pr_denormed = self._enforce_nonnegative_denormalized_variable(
                preds[self._pr_name], month_of_batch, self._pr_name
            )
            if self.hparams.nonnegativity_at_train_time:
                preds[self._pr_name] = tmp_pr_normed
        else:
            # Only get the denormalized scale variables (but do not enforce non-negativity)
            pr_denormed = self._denormalize_variable(preds[self._pr_name], month_of_batch, self._pr_name)
        ps_denormed = self._denormalize_variable(preds[self._ps_name], month_of_batch, self._ps_name)

        # Compute main loss by output variable e.g. output_name = 'pr_pre', 'tas_pre', etc.
        loss = 0.0
        for output_name, output_pred in preds.items():
            loss_var = self.criterion[output_name](output_pred, Y[output_name])
            train_log[f'{output_name}/train/loss'] = loss_var.item()
            loss += self.loss_weights[output_name] * loss_var
        train_log['train/loss_mse'] = loss.item()  # log the main MSE loss (without physics losses)

        # Compute physics loss (if any)
        physics_loss, train_log = self._physics_constraints_loss(
            input_tensor=X,
            month_of_batch=month_of_batch,
            precip_denormed=pr_denormed,
            pressure_denormed=ps_denormed,
            logging_dict=train_log
        )
        # Add physics loss to the main loss
        loss += physics_loss
        # Logging of train loss and other diagnostics
        train_log["train/loss"] = loss.item()

        # Count number of zero gradients as diagnostic tool
        train_log['n_zero_gradients'] = sum(
            [int(torch.count_nonzero(p.grad == 0))
             for p in self.parameters() if p.grad is not None
             ]) / self.n_params

        self.log_dict(train_log, prog_bar=False)
        return {"loss": loss}  # , "targets": Y, "preds": preds)}   # detach preds if they are returned!

    def _physics_constraints_loss(self,
                                  input_tensor: Tensor,
                                  month_of_batch: Tensor,
                                  precip_denormed: Tensor,
                                  pressure_denormed: Tensor,
                                  logging_dict: dict = None
                                  ) -> (float, Dict[str, float]):
        physics_loss = 0.0
        logging_dict = logging_dict or dict()
        # Soft loss constraints:
        # First, some auxiliary variable loading (e.g. climatologies, evaporation, ...)
        batch_monthly_clim_errs = {
            clim_err: self._get_monthly_data_for_batch(getattr(self, clim_err), month_of_batch, dim=0)
            for clim_err in ['PE_clim_err', 'PS_clim_err', 'Precip_clim_err']
        }
        aux_vars_denormed: Dict[str, Tensor] = dict()
        for aux_var in self.AUX_VARS:
            aux_var_idx = self.input_var_to_idx[aux_var]  # e.g. index of evspsbl_pre in the X tensor
            aux_vars_denormed[stem_var_id(aux_var)] = self._denormalize_variable(
                input_tensor[..., aux_var_idx], month_of_batch, output_var_name=aux_var
            )
        # -------------> Constraint 2 - Precipitation energy budget
        physics_loss2 = precipitation_energy_budget_constraint(
            precipitation=precip_denormed,
            sea_surface_heat_flux=aux_vars_denormed['hfss'],
            toa_sw_net_radiation=aux_vars_denormed['netTOARad'],
            surface_lw_net_radiation=aux_vars_denormed['netSurfRad'],
            PR_Err=batch_monthly_clim_errs['Precip_clim_err']
        )
        physics_loss2_abs = torch.abs(physics_loss2).mean()
        logging_dict['train/physics/loss2'] = physics_loss2.mean().item()
        logging_dict['train/physics/loss2_abs'] = physics_loss2_abs.item()
        if self.hparams.physics_loss_weights[1] > 0:
            physics_loss += self.hparams.physics_loss_weights[1] * 0.1 * physics_loss2_abs

        # -------------> Constraint 3 - Global moisture constraint
        # Compute the soft loss:
        physics_loss3 = global_moisture_constraint(
            precipitation=precip_denormed,
            evaporation=aux_vars_denormed['evspsbl'],
            PE_err=batch_monthly_clim_errs['PE_clim_err']
        )
        physics_loss3_abs = torch.abs(physics_loss3).mean()
        logging_dict['train/physics/loss3'] = physics_loss3.mean().item()
        logging_dict['train/physics/loss3_abs'] = physics_loss3_abs.item()
        if self.hparams.physics_loss_weights[2] > 0:
            # add the (weighted) loss to the main loss
            physics_loss += self.hparams.physics_loss_weights[2] * physics_loss3_abs

        # -------------> Constraint 5 - Mass conservation constraint
        # Compute the soft loss
        physics_loss5 = mass_conservation_constraint(
            surface_pressure=pressure_denormed,
            PS_err=batch_monthly_clim_errs['PS_clim_err']
        )
        physics_loss5_abs = torch.abs(physics_loss5).mean()
        logging_dict['train/physics/loss5'] = physics_loss5.mean().item()
        logging_dict['train/physics/loss5_abs'] = physics_loss5_abs.item()
        if self.hparams.physics_loss_weights[4] > 0:
            # add the (weighted) loss to the main loss (divide by 100_000 to allow loss weight to be around 1)
            physics_loss += self.hparams.physics_loss_weights[4] * physics_loss5_abs / 100_000
        return physics_loss, logging_dict

    def _enforce_nonnegative_denormalized_variable(self, *args, **kwargs) -> (Tensor, Tensor):
        """
        Enforce that the denormalized-scale variable is nonnegative

        Returns:
            A tuple of (normalized-scale variable, denormalized-scale variable), where both tensors will not have
            nonnegative values (the denormalized-scale tensor as-is, the normalized-scale one when denormalized).
        """
        # First, bring the variable to its original scale (i.e. denormalize it)
        denormed_var, monthly_mean, monthly_std = self._denormalize_variable(*args, **kwargs, return_monthly_stats=True)
        # enforce nonnegative values in the original/raw-scale
        nonnegative_var = nonnegative_precipitation(denormed_var)
        # bring back variable to the normalized scale
        normed_nonnegative_var = standardize(nonnegative_var, monthly_mean, monthly_std)
        return normed_nonnegative_var, nonnegative_var

    def training_epoch_end(self, outputs: List[Any]):
        train_time = time.time() - self._start_epoch_time
        self.log_dict({'epoch': float(self.current_epoch), "time/train": train_time})

    # --------------------- evaluation with PyTorch Lightning
    def _evaluation_per_variable(self,
                                 preds: Dict[str, Tensor],
                                 targets: Dict[str, Tensor],
                                 torch_metrics: nn.ModuleDict,
                                 manually_call_update: bool = False,
                                 ) -> Dict[str, float]:
        log_dict = dict()
        for metric_name, metric in torch_metrics.items():
            out_var_in_metric_name = [ovn for ovn in self.logging_output_var_names if ovn in metric_name]
            if len(out_var_in_metric_name) == 0:
                # if the metric is not related to any specific output variable (e.g. val/mse)
                continue
            elif len(out_var_in_metric_name) == 1:
                # metric can be easily identified by the output variable name
                out_var_name = out_var_in_metric_name[0]
            elif len(out_var_in_metric_name) > 1:
                # Multiple options: take the var with maximum character overlap with the metric name (pr_pre vs pr)
                out_var_name = max(out_var_in_metric_name, key=lambda ovn: len(ovn))
            # out_var_name = metric_name.split('/')[0] if '_pre' in metric_name else metric_name.split('/')[1]
            # compute the metric
            if manually_call_update:
                metric.update(preds[out_var_name], targets[out_var_name])
            else:
                metric(preds[out_var_name], targets[out_var_name])
                # save the metric value
                log_dict[metric_name] = metric
        return log_dict

    def _evaluation_step(self,
                         batch: Any, batch_idx: int,
                         torch_metrics: Optional[nn.ModuleDict] = None,
                         **kwargs):
        X, Y = batch
        preds = self._raw_predict_single_tensor(X)
        log_dict1 = dict()
        kwargs['sync_dist'] = True  # for DDP training
        # First compute the bulk error metrics (averaging out over all target variables)
        for metric_name, metric in torch_metrics.items():
            if any([out_var_name in metric_name for out_var_name in self.logging_output_var_names]):
                # Per output variable error will not be computed yet
                continue
            metric(preds, Y)  # compute metrics (need to be in separate line to the following line!)
            log_dict1[metric_name] = metric
        self.log_dict(log_dict1, on_step=True, on_epoch=True, **kwargs)  # log metric objects

        # Now compute per output variable errors
        preds = self.raw_outputs_to_denormalized_per_variable_dict(preds, input_tensor=X,
                                                                   return_normalized_outputs=True)
        Y = self.postprocess_raw_predictions(Y, input_tensor=X, return_normalized_outputs=True)
        log_dict2 = self._evaluation_per_variable(preds, Y, torch_metrics)
        kwargs['prog_bar'] = False  # do not show in progress bar the per-output variable metrics
        self.log_dict(log_dict2, on_step=False, on_epoch=True, **kwargs)  # log metric objects
        return {'targets': Y, 'preds': preds}

    def _evaluation_get_preds(self, outputs: List[Any]) -> Dict[str, np.ndarray]:
        targets = torch.cat([batch['targets'] for batch in outputs], dim=0).cpu().numpy()
        preds = torch.cat([batch['preds'] for batch in outputs], dim=0).detach().cpu().numpy()
        return {'targets': targets, 'preds': preds}

    def on_validation_epoch_start(self) -> None:
        self._start_validation_epoch_time = time.time()

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        results = self._evaluation_step(batch, batch_idx, torch_metrics=self.val_metrics, prog_bar=True)
        return results

    def validation_epoch_end(self, outputs: List[Any]) -> dict:
        val_time = time.time() - self._start_validation_epoch_time
        val_stats = {"time/validation": val_time}
        # validation_outputs = self._evaluation_get_preds(outputs)
        # Y_val, validation_preds = validation_outputs['targets'], validation_outputs['preds']

        # target_val_metric = val_stats.pop(self.hparams.monitor, None)
        self.log_dict({**val_stats, 'epoch': float(self.current_epoch)}, prog_bar=False)
        # Show the main validation metric on the progress bar:
        # self.log(self.hparams.monitor, target_val_metric, prog_bar=True)
        return val_stats

    def on_test_epoch_start(self) -> None:
        self._start_test_epoch_time = time.time()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        results = self._evaluation_step(batch, batch_idx, torch_metrics=self.test_metrics)
        return results

    def test_epoch_end(self, outputs: List[Any]):
        test_time = time.time() - self._start_test_epoch_time
        self.log("time/test", test_time)

    def on_predict_start(self) -> None:
        assert self.trainer.datamodule._data_predict is not None, "_data_predict is None"
        assert self.trainer.datamodule._data_predict.dataset_id == 'predict', "dataset_id is not 'predict'"
        for pdl in self.trainer.predict_dataloaders:
            assert pdl.dataset.dataset_id == 'predict', f"dataset_id is not 'predict', but {pdl.dataset.dataset_id}"

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None, **kwargs
                     ) -> Dict[str, Dict[str, Tensor]]:
        X, Y = batch
        # directly predict full original-scale outputs (separated per output variable into a dictionary)
        preds: Dict[str, Tensor] = self.predict(X)
        # Only split the targets into per-output variable tensors + use the correct output var names (without '_pre')
        Y = {stem_var_id(k): v for k, v in self._split_raw_preds_per_target_variable(Y).items()}
        # evaluate the predictions vs the original-scale targets
        _ = self._evaluation_per_variable(preds, Y, self.predict_metrics, manually_call_update=True)
        # Not possible in predict:
        # self.log_dict(log_dict, on_step=False, on_epoch=True, **kwargs)  # log metric objects
        return {'targets': Y, 'preds': preds}

    def on_predict_end(self, results: List[Any] = None) -> None:
        if wandb.run is not None:
            log_dict = {'epoch': float(self.current_epoch)}
            for k, v in self.predict_metrics.items():
                log_dict[k] = float(v.compute().item())
                v.reset()
            self.log_text.info(log_dict)
            print(log_dict, wandb.run.id)
            wandb.log(log_dict)
        else:
            self.log_text.warning("Wandb is not initialized, so no predictions are logged")

    # ---------------------------------------------------------------------- Optimizers and scheduler(s)
    def _get_optim(self, optim_name: str, **kwargs):
        """
        Method that returns the torch.optim optimizer object.
        May be overridden in subclasses to provide custom optimizers.
        """
        return torch.optim.AdamW(self.parameters(), **kwargs)
        from timm.optim import create_optimizer_v2
        return create_optimizer_v2(model_or_params=self, opt=optim_name, **kwargs)

    def configure_optimizers(self):
        """ Configure optimizers and schedulers """
        if 'name' not in to_DictConfig(self.hparams.optimizer).keys():
            self.log_text.info(" No optimizer was specified, defaulting to AdamW with 1e-4 lr.")
            self.hparams.optimizer.name = 'adamw'

        if hasattr(self, 'no_weight_decay'):  # e.g. for positional embeddings that shouldn't be regularized
            self.log_text.info(" Model has method no_weight_decay, which will be used.")
        optim_kwargs = {k: v for k, v in self.hparams.optimizer.items() if k not in ['name', '_target_']}
        optimizer = self._get_optim(self.hparams.optimizer.name, **optim_kwargs)
        # Build the scheduler
        if self.hparams.scheduler is None:
            return optimizer  # no scheduler
        else:
            if '_target_' not in to_DictConfig(self.hparams.scheduler).keys():
                raise ValueError("Please provide a _target_ class for model.scheduler arg!")
            scheduler_params = to_DictConfig(self.hparams.scheduler)
            scheduler = hydra.utils.instantiate(scheduler_params, optimizer=optimizer)

        if not hasattr(self.hparams, 'monitor') or self.hparams.monitor is None:
            self.hparams.monitor = f'val/mse'
        if not hasattr(self.hparams, 'mode') or self.hparams.mode is None:
            self.hparams.mode = 'min'

        lr_dict = {'scheduler': scheduler, 'monitor': self.hparams.monitor}  # , 'mode': self.hparams.mode}
        return {'optimizer': optimizer, 'lr_scheduler': lr_dict}

    # Unimportant methods
    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def register_buffer_dummy(self, name, tensor, **kwargs):
        try:
            self.register_buffer(name, tensor, **kwargs)
        except TypeError:  # old pytorch versions do not have the arg 'persistent'
            self.register_buffer(name, tensor)

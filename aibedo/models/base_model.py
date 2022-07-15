import logging
import time
from typing import Optional, List, Any, Dict, Sequence, Union

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from timm.optim import create_optimizer_v2
from torch import Tensor, nn
from pytorch_lightning import LightningModule
import torchmetrics
from aibedo.data_transforms.normalization import NormalizationMethod

from aibedo.data_transforms.transforms import AbstractTransform
from aibedo.utilities.utils import get_logger, to_DictConfig, get_loss
from aibedo.skeleton_framework.spherical_unet.utils.samplings import icosahedron_nodes_calculator


class BaseModel(LightningModule):
    """
    This is a template class, that should be inherited by any AIBEDO stand-alone ML model.
    Methods that need to be implemented by your concrete NN model (just as if you would define a torch.nn.Module):
        - __init__(.)
        - forward(.)

    The other methods may be overridden as needed.
    It is recommended to define the attribute
        - self.example_input_array = torch.randn(<YourModelInputShape>)  # batch dimension can be anything, e.g. 7

    Args:
        datamodule_config: DictConfig with the configuration of the datamodule
        input_transform: AbstractTransform with an optional input transform
        optimizer: DictConfig with the optimizer configuration (e.g. for AdamW)
        scheduler: DictConfig with the scheduler configuration (e.g. for CosineAnnealingLR)
        monitor: str with the name of the metric to monitor, e.g. 'val/mse'
        mode: str with the mode of the monitor, e.g. 'min' (lower is better)
        loss_function: str with the name of the loss function, e.g. 'mean_squared_error'
        name: optional str with a name for the model
        verbose: Whether to print logs or not
    ------------
    Read the docs regarding LightningModule for more information.:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(self,
                 datamodule_config: DictConfig = None,
                 input_transform: Optional[AbstractTransform] = None,
                 optimizer: Optional[DictConfig] = None,
                 scheduler: Optional[DictConfig] = None,
                 monitor: Optional[str] = None,
                 mode: str = "min",
                 loss_weights: Union[Sequence[float], Dict[str, float]] = (0.33, 0.33, 0.33),
                 physics_loss_weights: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0),
                 loss_function: str = "mean_squared_error",
                 output_normalizer: Optional[Dict[str, NormalizationMethod]] = None,
                 name: str = "",
                 verbose: bool = True,
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

        if input_transform is None or isinstance(input_transform, AbstractTransform):
            self.input_transform = input_transform
        else:
            self.input_transform: AbstractTransform = hydra.utils.instantiate(input_transform)

        self._data_dir = self._output_vars = None
        if datamodule_config is not None:
            # Infer the data dimensions
            self.spatial_dim = n_pixels = icosahedron_nodes_calculator(datamodule_config.order)
            self._num_input_features = in_channels = len(datamodule_config.input_vars)
            self._num_output_features = out_channels = len(datamodule_config.output_vars)
            self._output_vars = datamodule_config.output_vars

        self.output_normalizer = output_normalizer

        # loss function (one per target variable)
        self.criterion = {v: get_loss(loss_function, reduction='mean') for v in self.output_var_names}

        # Timing variables to track the training/epoch/validation time
        self._start_validation_epoch_time = self._start_test_epoch_time = self._start_epoch_time = None

        # Metrics
        self.val_metrics = nn.ModuleDict({
            'val/mse': torchmetrics.MeanSquaredError(squared=True),
            **{
                f'{output_var}/val/mse': torchmetrics.MeanSquaredError(squared=True)
                for output_var in self.output_var_names
            }
        })
        self._test_metrics = None
        # Check that the args/hparams are valid
        self._check_args()

    @property
    def num_input_features(self) -> int:
        return self._num_input_features

    @property
    def num_output_features(self) -> int:
        return self._num_output_features

    @property
    def test_set_name(self) -> str:
        return self.trainer.datamodule.test_set_name if hasattr(self.trainer.datamodule, 'test_set_name') else 'test'

    @property
    def test_metrics(self):
        if self._test_metrics is None:
            self._test_metrics = nn.ModuleDict({
                f'{self.test_set_name}/mse': torchmetrics.MeanSquaredError(squared=True),
                **{
                    f'{output_var}/{self.test_set_name}/mse': torchmetrics.MeanSquaredError(squared=True)
                    for output_var in self.output_var_names
                }
            }).to(self.device)
        return self._test_metrics

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
    def output_var_names(self) -> List[str]:
        if self._output_vars is None:
            self._output_vars = self.trainer.datamodule.hparams.output_vars
        return self._output_vars

    @property
    def output_normalizer(self) -> Dict[str, NormalizationMethod]:
        return self._output_normalizer

    @output_normalizer.setter
    def output_normalizer(self, output_normalizer: Dict[str, NormalizationMethod]):
        self._output_normalizer = output_normalizer
        if self.output_normalizer is None:
            self.log_text.info(' No output normalization for outputs is used.')
        else:
            normalizer_name = list(self.output_normalizer.values())[0].__class__.__name__
            self.log_text.info(f' Using output normalization "{normalizer_name}" for prediction.')
            for v in self.output_normalizer.values():
                v.change_input_type(torch.Tensor)

    def _check_args(self):
        """Check if the arguments are valid."""
        plw = self.hparams.physics_loss_weights
        if len(plw) != 5:
            raise ValueError(f'The number of physics loss weights must be 5, but got {plw}')
        if plw[3] > 0 and plw[3] != 1:
            self.log_text.info(f'The fourth physics loss weight must be 0 or 1, but got {plw[3]}. Setting it to 1.')
            self.hparams.physics_loss_weights[3] = 1
        lw = self.hparams.loss_weights
        if isinstance(lw, dict) and len(lw.keys()) != len(self.output_var_names) or \
                isinstance(lw, Sequence) and len(lw) != len(self.output_var_names):
            raise ValueError(f'The number of loss weights must be same as #output-vars={len(self.output_var_names)}'
                             f', but got {lw}')
        if isinstance(lw, Sequence):
            self.hparams.loss_weights = {v: lw[i] for i, v in enumerate(self.output_var_names)}

    def forward(self, X):
        """
        Downstream model forward pass, input X will be the (batched) output from self.input_transform
        """
        raise NotImplementedError('Base model is an abstract class!')

    def raw_predict(self, X) -> Tensor:
        Y = self(X)  # forward pass
        return Y

    def _split_raw_preds_per_target_variable(self, predictions: Tensor) -> Dict[str, Tensor]:
        preds_per_target_variable = {
            var_name: predictions[..., i]
            for i, var_name in enumerate(self.output_var_names)
        }
        return preds_per_target_variable

    def predict(self, X) -> Dict[str, Tensor]:
        Y_normed = self.raw_predict(X)
        Y = self._split_raw_preds_per_target_variable(Y_normed)
        # TODO: implement the output de-normalization
        return Y

    def _raw_to_full_preds(self,
                           flux_profile_pred: Dict[str, Tensor],
                           ) -> Dict[str, Tensor]:
        full_preds = dict()
        for flux_type, flux_pred in flux_profile_pred.items():
            if self.output_normalizer is not None:
                flux_pred = self.output_normalizer[flux_type].inverse_normalize(flux_pred)
        return full_preds

    # --------------------- training with PyTorch Lightning
    def on_train_start(self) -> None:
        """ Log some info about the model/data at the start of training """
        self.log('Parameter count', float(self.n_params))
        self.log('Training set size', float(len(self.trainer.datamodule._data_train)))
        self.log('Validation set size', float(len(self.trainer.datamodule._data_val)))

        if self._output_normalizer is None and self.trainer.datamodule.normalizer is not None:
            self.log_text.info(" Dynamically adding the output normalizer from trainer.datamodule")
            self.output_normalizer = self.trainer.datamodule.normalizer.output_normalizer

    def on_train_epoch_start(self) -> None:
        self._start_epoch_time = time.time()

    def train_step_initial_log_dict(self) -> dict:
        return dict()

    def training_step(self, batch: Any, batch_idx: int):
        X, Y = batch
        train_log = self.train_step_initial_log_dict()

        # Predict normalized outputs and split them into output_var: Tensor preds/targets of that var
        preds = self._split_raw_preds_per_target_variable(self.raw_predict(X))
        Y = self._split_raw_preds_per_target_variable(Y)

        loss = 0.0
        # Compute loss by output variable
        for output_name, output_pred in preds.items():
            loss_var = self.criterion[output_name](output_pred, Y[output_name])
            loss += self.hparams.loss_weights[output_name] * loss_var
            train_log[f'{output_name}/train/loss'] = loss_var.item()

        # Logging of train loss and other diagnostics
        train_log["train/loss"] = loss.item()

        train_log['n_zero_gradients'] = sum(  # Count number of zero gradients as diagnostic tool
            [int(torch.count_nonzero(p.grad == 0))
             for p in self.parameters() if p.grad is not None
             ]) / self.n_params

        self.log_dict(train_log, prog_bar=False)
        return {"loss": loss}  # , "targets": Y, "preds": preds)}   # detach preds!

    def training_epoch_end(self, outputs: List[Any]):
        train_time = time.time() - self._start_epoch_time
        self.log_dict({'epoch': float(self.current_epoch), "time/train": train_time})

    # --------------------- evaluation with PyTorch Lightning
    def _evaluation_step(self,
                         batch: Any, batch_idx: int,
                         torch_metrics: Optional[nn.ModuleDict] = None,
                         **kwargs):
        X, Y = batch
        preds = self.raw_predict(X)
        log_dict = dict()
        # First compute the bulk error metrics (averaging out over all target variables)
        for metric_name, metric in torch_metrics.items():
            if any([out_var_name in metric_name for out_var_name in self.output_var_names]):
                # Per output variable error are not computed yet
                continue
            metric(preds, Y)  # compute metrics (need to be in separate line to the following line!)
            log_dict[metric_name] = metric
        self.log_dict(log_dict, on_step=True, on_epoch=True, **kwargs)  # log metric objects
        # Now compute per output variable errors
        log_dict = dict()
        preds = self._split_raw_preds_per_target_variable(preds)
        Y = self._split_raw_preds_per_target_variable(Y)
        for metric_name, metric in torch_metrics.items():
            if not any([out_var_name in metric_name for out_var_name in self.output_var_names]):
                continue
            out_var_name = metric_name.split('/')[0]
            metric(preds[out_var_name], Y[out_var_name])
            log_dict[metric_name] = metric
        kwargs['prog_bar'] = False   # do not show in progress bar the per-output variable metrics
        self.log_dict(log_dict, on_step=True, on_epoch=True, **kwargs)  # log metric objects
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

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        return self._evaluation_step(batch, batch_idx)

    # ---------------------------------------------------------------------- Optimizers and scheduler(s)
    def _get_optim(self, optim_name: str, **kwargs):
        """
        Method that returns the torch.optim optimizer object.
        May be overridden in subclasses to provide custom optimizers.
        """
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

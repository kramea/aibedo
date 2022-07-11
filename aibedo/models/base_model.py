import logging
import time
from typing import Optional, List, Any, Dict

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

        if datamodule_config is not None:
            # Infer the data dimensions
            self.spatial_dim = n_pixels = icosahedron_nodes_calculator(datamodule_config.order)
            self._num_input_features = in_channels = len(datamodule_config.input_vars)
            self._num_output_features = out_channels = len(datamodule_config.output_vars)

        self.output_normalizer = output_normalizer

        # loss function
        self.criterion = get_loss(loss_function, reduction='mean')
        # Timing variables to track the training/epoch/validation time
        self._start_validation_epoch_time = self._start_test_epoch_time = self._start_epoch_time = None
        # Metrics
        # self.train_mse = torchmetrics.MeanSquaredError(squared=True)
        self.val_metrics = nn.ModuleDict({
                'val/mse': torchmetrics.MeanSquaredError(squared=True),
        })
        self._test_metrics = None

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
            }).to(self.device)
        return self._test_metrics

    @property
    def n_params(self):
        """ Returns the number of parameters in the model """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def output_normalizer(self) -> Dict[str, NormalizationMethod]:
        return self._output_normalizer

    @property
    def data_dir(self) -> str:
        if not hasattr(self, '_data_dir') or self._data_dir is None:
            self._data_dir = self.trainer.datamodule.hparams.data_dir
        return self._data_dir

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

    def forward(self, X):
        """
        Downstream model forward pass, input X will be the (batched) output from self.input_transform
        """
        raise NotImplementedError('Base model is an abstract class!')

    def raw_predict(self, X) -> Dict[str, Tensor]:
        Y = self(X)  # Y might be in normalized scale or not
        Y = self._split_raw_preds_per_target_variable(Y)
        return Y

    def _split_raw_preds_per_target_variable(self, predictions: Tensor) -> Dict[str, Tensor]:
        # flux_profile_pred = self.output_postprocesser.split_vector_by_variable(predictions)
        # return flux_profile_pred
        return predictions

    def predict(self, X) -> Dict[str, Tensor]:
        Y_normed_fluxes = self.raw_predict(X)
        return Y_normed_fluxes
        # full_prediction = self._raw_to_full_preds(Y_normed_fluxes)
        # return full_prediction

    def _raw_to_full_preds(self,
                           flux_profile_pred: Dict[str, Tensor],
                           ) -> Dict[str, Tensor]:
        full_preds = dict()
        for flux_type, flux_pred in flux_profile_pred.items():
            if self.output_normalizer is not None:
                flux_pred = self.output_normalizer[flux_type].inverse_normalize(flux_pred)
        return full_preds

    def _apply(self, fn):
        super(BaseModel, self)._apply(fn)
        if self.output_normalizer is not None:
            self.output_normalizer.apply_torch_func(fn)
        return self

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

        if self.output_normalizer is None:
            # Directly predict full/raw/non-normalized outputs
            preds = self.predict(X)
        else:
            # Predict normalized outputs
            preds = self.raw_predict(X)

        loss = self.criterion(preds, Y)

        train_log = self.train_step_initial_log_dict()
        train_log["train/loss"] = loss.item()

        train_log['n_zero_gradients'] = sum(    # Count number of zero gradients as diagnostic tool
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
        preds = self.predict(X)
        log_dict = dict()
        for metric_name, metric in torch_metrics.items():
            metric(preds, Y)  # compute metrics (need to be in separate line to the following line!)
            log_dict[metric_name] = metric
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

        if hasattr(self, 'no_weight_decay'):   # e.g. for positional embeddings that shouldn't be regularized
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

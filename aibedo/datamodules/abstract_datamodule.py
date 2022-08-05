import itertools
import logging
import os
from os.path import join
from typing import Optional, List, Callable, Sequence, Dict
from omegaconf import DictConfig
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import numpy as np
import xarray as xr

from aibedo.constants import CLIMATE_MODELS_ALL
from aibedo.datamodules.torch_dataset import AIBEDOTensorDataset
from aibedo.models.base_model import BaseModel
from aibedo.utilities.constraints import AUXILIARY_VARS
from aibedo.utilities.naming import var_names_to_clean_name
from aibedo.utilities.utils import get_logger, raise_error_if_invalid_value

log = get_logger(__name__)


class AIBEDO_DataModule(pl.LightningDataModule):
    """
    ----------------------------------------------------------------------------------------------------------
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """
    _data_train: AIBEDOTensorDataset
    _data_val: AIBEDOTensorDataset
    _data_test: AIBEDOTensorDataset
    _data_predict: AIBEDOTensorDataset

    def __init__(self,
                 input_vars: Sequence[str],
                 output_vars: Sequence[str],
                 data_dir: str,
                 input_filename: str = "compress.isosph5.CESM2.historical.r1i1p1f1.Input.Exp8_fixed.nc",
                 time_lag: int = 0,
                 partition: Sequence[float] = (0.8, 0.1, 0.1),
                 prediction_data: str = "same_as_test",
                 model_config: DictConfig = None,
                 batch_size: int = 64,
                 eval_batch_size: int = 512,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 verbose: bool = True,
                 seed: int = 43,
                 ):
        """
        Args:
            input_vars: list of input variables/predictors, e.g. ['crelSurf_pre', 'crel_pre', 'cresSurf_pre']
            output_vars: list of output/target variables, e.g. ['tas', 'pr', 'psl']
            data_dir (str):  A path to the data folder that contains the input and output files.
            partition (tuple): partition of the data into train, validation and test fractions/sets.
                                Train and validation (indices 0 and 1) must be floats.
                                Test (index 2) can be a float or a string.
                                   -> If test is a string, it must be one of the following: 'merra2', 'era5'
            batch_size (int): Batch size for the training dataloader
            eval_batch_size (int): Batch size for the test and validation dataloader's
            num_workers (int): Dataloader arg for higher efficiency
            pin_memory (bool): Dataloader arg for higher efficiency
        """
        super().__init__()
        # The following makes all args available as, e.g., self.hparams.batch_size
        self.save_hyperparameters(ignore=['model_config', 'prediction_data'])
        self.model_config = model_config
        self._data_train = self._data_val = self._data_test = self._data_predict = None
        self._possible_test_sets = ['merra2', 'era5']
        self._possible_prediction_sets = self._possible_test_sets + ['val', 'same_as_test'] + CLIMATE_MODELS_ALL
        self.prediction_data = prediction_data
        self.hparams.auxiliary_vars = AUXILIARY_VARS if model_config.use_auxiliary_vars else []
        self.window = model_config.window if hasattr(model_config, 'window') else 1
        input_var_names = [[f'{v}_mon{i}' for i in range(self.window)] for v in input_vars]
        self.input_var_names = list(itertools.chain(*input_var_names))  # flatten list of lists
        self.input_var_to_idx = {
            var: i for i, var
            in enumerate(self.input_var_names + ['month'] + self.hparams.auxiliary_vars)
        }
        self._var_names_to_clean_name = var_names_to_clean_name()
        self._set_geographical_metadata()
        self._check_args()

    @property
    def var_names_to_clean_name(self):
        return self._var_names_to_clean_name

    @property
    def files_id(self) -> str:
        return self.hparams.input_filename.split('.')[1]

    @property
    def month_index(self) -> int:
        # By default the month index is concatenated to the D input vars,
        # so that indices 0...D-1 are the input vars and D is the month
        return len(self.hparams.input_vars)

    @property
    def test_set_name(self) -> str:
        train_frac, val_frac, test_frac = self.hparams.partition
        if test_frac == 'merra2':
            return 'test/MERRA2'
        elif test_frac == 'era5':
            return 'test/ERA5'
        elif isinstance(test_frac, float):
            return f'test/{self._esm_name}'
        else:
            raise ValueError(f"Unknown test_frac: {test_frac}")

    @property
    def prediction_data(self) -> str:
        return self._prediction_data

    @prediction_data.setter
    def prediction_data(self, value: str):
        self._prediction_data = raise_error_if_invalid_value(value, self._possible_prediction_sets, 'prediction_data')
        self._data_predict = None

    @property
    def prediction_set_name(self) -> str:
        if self.prediction_data == 'same_as_test':
            return self.test_set_name.replace('test', 'predict')
        elif self.prediction_data == 'val':
            return f'predict/{self._esm_name}'
        elif self.prediction_data in self._possible_test_sets or self.prediction_data in CLIMATE_MODELS_ALL:
            return f'predict/{self.prediction_data.upper()}'
        else:
            raise ValueError(f"Unknown prediction data being used: {self.prediction_data}")

    def _check_args(self):
        """Check if the arguments are valid."""
        partition = self.hparams.partition
        if len(partition) != 3:
            raise ValueError(f"partition must be a tuple of 3 values, but got {partition}, type: {type(partition)}")
        test_frac = partition[2]
        if isinstance(test_frac, str):
            raise_error_if_invalid_value(test_frac, possible_values=self._possible_test_sets, name='partition[2]')
            if partition[0] + partition[1] != 1:
                self.hparams.partition = (partition[0], 1 - partition[0], partition[2])
                log.warning(
                    "partition[0] + partition[1] does not sum to 1 and test_frac is a string. partition[1] will be set to 1 - partition[0].")
        elif partition[0] + partition[1] + partition[2] != 1:
            raise ValueError(
                f"partition must sum to 1, but it sums to {partition[0] + partition[1] + partition[2]}")

    def _set_geographical_metadata(self):
        self._esm_name = self.hparams.input_filename.split('.')[2]

        input_file = os.path.join(self.hparams.data_dir, self.hparams.input_filename)
        inDS = xr.open_dataset(input_file)

        self.lon_list: np.ndarray = inDS.lon.values
        self.lat_list: np.ndarray = inDS.lat.values
        lsmask_round = [(round(x) if x == x else 0.5) for x in inDS.lsMask.values[0]]
        self.ls_mask = np.array([0 if x < 0 else 1 if x > 1 else x for x in lsmask_round])
        # print(f'LS mask shape: {inDS.lsMask.data.shape}, \n{inDS.lsMask.data[0]} '
        #      f'\n******************\n{inDS.lsMask.data[1]}')

    def tropics_mask(self) -> np.ndarray:
        # tropic = df3[(df3.Lat < 30) & (df3.Lat > -30)]
        return (-30 < self.lat_list) & (self.lat_list < 30)

    def mid_latitudes_mask(self) -> np.ndarray:
        # temperate = df2[((df2.Lat>30) & (df2.Lat<60)) | ((df2.Lat <-30 ) & (df2.Lat > -60)) ]
        return ((30 < self.lat_list) & (self.lat_list < 60)) | ((-60 < self.lat_list) & (self.lat_list < -30))

    def arctic_mask(self) -> np.ndarray:
        return self.lat_list > 60

    def antarctic_mask(self) -> np.ndarray:
        return self.lat_list < -60

    def land_mask(self) -> np.ndarray:
        return self.ls_mask == 1

    def sea_mask(self) -> np.ndarray:
        return self.ls_mask == 0

    def masks(self) -> Dict[str, np.ndarray]:
        return {
            'tropics': self.tropics_mask(),
            'mid_latitudes': self.mid_latitudes_mask(),
            'arctic': self.arctic_mask(),
            'antarctic': self.antarctic_mask(),
            'land': self.land_mask(),
            'sea': self.sea_mask(),
        }

    def _shared_dataloader_kwargs(self) -> dict:
        return dict(num_workers=int(self.hparams.num_workers), pin_memory=self.hparams.pin_memory)

    def _shared_eval_dataloader_kwargs(self) -> dict:
        return dict(**self._shared_dataloader_kwargs(), batch_size=self.hparams.eval_batch_size, shuffle=False)

    def train_dataloader(self):
        return DataLoader(
            dataset=self._data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            **self._shared_dataloader_kwargs(),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self._data_val, **self._shared_eval_dataloader_kwargs()
        ) if self._data_val is not None else None

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self._data_test, **self._shared_eval_dataloader_kwargs())

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(dataset=self._data_predict, **self._shared_eval_dataloader_kwargs())

    def get_predictions(self, model: BaseModel,
                        dataloader: DataLoader = None,
                        device: torch.device = None,
                        **prediction_kwargs
                        ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get the predictions and groundtruth for the prediction set (self._data_predict), by default the test data.

        Args:
            model: The model to use for prediction.
            dataloader: The (optional) dataloader to use for prediction. By default, the predict_dataloader is used.
            device: The device ('cuda', 'cpu', etc.)

        Returns:
            A dictionary {'preds': predictions, 'targets': ground_truth}
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model: BaseModel = model.to(device)
        predict_loader = self.predict_dataloader() if dataloader is None else dataloader
        if predict_loader.dataset is None:
            if dataloader is not None:
                log.warning(f" The dataloader provided has no dataset. Using the default predict_dataloader instead.")
            self.setup(stage='predict')
            predict_loader = self.predict_dataloader()

        if predict_loader.dataset.dataset_id == 'predict':
            log.info(' Assuming that the used dataloader has raw/non-normalized targets.')
            RAW_TARGETS = True
        else:
            RAW_TARGETS = False

        preds, targets = dict(), dict()
        for i, batch in enumerate(predict_loader):
            data_in, data_out = batch
            # get targets/preds dict of output var name -> tensor (in denormalized scale!)
            MONTH_IDX = self.input_var_to_idx['month']
            month_of_outputs = data_in[:, 0, MONTH_IDX]  # idx 0 is arbitrary; has shape (batch_size,)

            prediction_kwargs['month_of_outputs'] = month_of_outputs
            batch_preds = model.predict(data_in.to(device), **prediction_kwargs)
            if RAW_TARGETS:
                # only split the targets by output variable
                batch_targets = model._split_raw_preds_per_target_variable(data_out)
                # but, need to use the correct output var keys/names (without '_pre'):
                batch_targets = {k.replace('_pre', ''): v for k, v in batch_targets.items()}
            else:
                # also denormalize the targets
                batch_targets = model.raw_outputs_to_denormalized_per_variable_dict(data_out, **prediction_kwargs)
                # batch_targets = {k: v.detach().cpu().numpy() for k, v in batch_targets.items()}
            # Now concatenate the predictions and the targets across all batches
            for out_var in batch_targets.keys():
                batch_preds_numpy = batch_preds[out_var].detach().cpu().numpy()
                batch_gt_numpy = batch_targets[out_var].detach().cpu().numpy()
                if i == 0:
                    preds[out_var] = batch_preds_numpy
                    targets[out_var] = batch_gt_numpy
                else:
                    preds[out_var] = np.concatenate((preds[out_var], batch_preds_numpy), axis=0)
                    targets[out_var] = np.concatenate((targets[out_var], batch_gt_numpy), axis=0)

        return {'preds': preds, 'targets': targets}

    def get_predictions_xarray(self, model: nn.Module,
                               variables='all',
                               also_targets: bool = True,
                               also_errors: bool = False,
                               return_normalized_outputs: bool = False,
                               **prediction_kwargs
                               ) -> xr.Dataset:
        numpy_preds_targets = self.get_predictions(model, **prediction_kwargs, return_normalized_outputs=return_normalized_outputs)
        preds, targets = numpy_preds_targets['preds'], numpy_preds_targets['targets']
        var_shape = preds[list(preds.keys())[0]].shape[:-1]
        dim_names = ['snapshot', 'latitude', 'longitude'] if len(var_shape) == 3 else ['snapshot', 'spatial_dim']

        data_vars = dict()
        if variables == 'all':
            out_vars = list(preds.keys())
        else:
            out_vars = list(variables)
            if any(['_pre' in v for v in out_vars]) and not return_normalized_outputs:
                raise ValueError(f"The variables {out_vars} contain vars with _pre (normalized)"
                                 f" but `return_normalized_outputs` is False.")
        for i, output_var in enumerate(out_vars):  # usually ['tas', 'ps', 'pr'], maybe also 'tas_pre' etc.
            output_var_pred = preds[output_var]
            output_var_target = targets[output_var]
            data_vars[f"{output_var}_preds"] = (dim_names, output_var_pred)
            if also_targets:
                data_vars[f"{output_var}_targets"] = (dim_names, output_var_target)
            if also_errors:
                diff = output_var_pred - output_var_target
                mae = np.abs(diff)
                data_vars[f'{output_var}_bias'] = (dim_names, diff)
                data_vars[f'{output_var}_mae'] = (dim_names, mae)
                data_vars[f'{output_var}_mae_score'] = (dim_names, mae / np.mean(output_var_target, axis=0))
        # data_vars['lat_list'] = (['latitude'], self.lat_list)
        # data_vars['lon_list'] = (['longitude'], self.lon_list)
        xr_dset = xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                longitude=(['spatial_dim'], self.lon_list),
                latitude=(['spatial_dim'], self.lat_list),
                #   longitude=self.lon_list,
                #   latitude=self.lat_list,
                #   spatial_dim=(('longitude', 'latitude'), [(x, y) for x, y in zip(self.lon_list, self.lat_list)]),
                snapshot=range(var_shape[0]),
                #  **self.masks(),
            ), attrs=dict(
                description=f"ML emulated predictions.",
                variable_names=";".join(out_vars),
            ))
        return xr_dset


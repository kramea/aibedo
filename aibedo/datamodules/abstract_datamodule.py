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
from aibedo.utilities.naming import var_names_to_clean_name
from aibedo.utilities.utils import get_logger

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

    def __init__(self,
                 input_vars: Sequence[str],
                 output_vars: Sequence[str],
                 data_dir: str,
                 input_filename: str = "compress.isosph5.CESM2.historical.r1i1p1f1.Input.Exp8_fixed.nc",
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
            data_dir (str):  A path to the data folder that contains the input and output files.
            batch_size (int): Batch size for the training dataloader
            eval_batch_size (int): Batch size for the test and validation dataloader's
            num_workers (int): Dataloader arg for higher efficiency
            pin_memory (bool): Dataloader arg for higher efficiency
        """
        super().__init__()
        # The following makes all args available as, e.g., self.hparams.batch_size
        self.save_hyperparameters(ignore=['model_config'])
        self.model_config = model_config
        self._data_train = self._data_val = self._data_test = self._data_predict = None
        self._possible_test_sets = ['merra2', 'era5']
        if self.model_config.physics_loss_weights[2] > 0 or True:
            self.hparams.auxiliary_vars = ['evspsbl_pre']
        else:
            self.hparams.auxiliary_vars = []
        self.input_var_to_idx = {
            var: i for i, var
            in enumerate(
                list(self.hparams.input_vars) + ['month'] + self.hparams.auxiliary_vars
         )}
        self._var_names_to_clean_name = var_names_to_clean_name()
        self._set_geographical_metadata()

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

    def get_predictions(self, model: nn.Module, dataloader: DataLoader = None,
                        filename: str = None, device: torch.device = None):
        """
        Get the predictions and groundtruth for the prediction set (self._data_predict), by default the test data.
        if filename is a string:
            Save the predictions and the ground truth to a numpy file (.npz).
            The saved file will have the following structure:
                - predictions: numpy array of the predictions
                - groundtruth: numpy array of the corresponding ground truth/targets

        Args:
            model: The model to use for prediction.
            dataloader: The (optional) dataloader to use for prediction. By default, the predict_dataloader is used.
            filename: The filepath to save the numpy file to.
            device: The device ('cuda', 'cpu', etc.)

        Returns:
            A dictionary {'preds': predictions, 'targets': ground_truth}
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        predict_loader = self.predict_dataloader() if dataloader is None else dataloader
        if predict_loader.dataset is None:
            if dataloader is not None:
                log.warning(f" The dataloader provided has no dataset. Using the default predict_dataloader instead.")
            self.setup(stage='predict')
            predict_loader = self.predict_dataloader()

        for i, batch in enumerate(predict_loader):
            data_in, data_out = batch
            preds = model(data_in.to(device))
            preds_numpy = preds.detach().cpu().numpy()
            gt_numpy = data_out.detach().cpu().numpy()
            if i == 0:
                predictions = preds_numpy
                groundtruth = gt_numpy
            else:
                predictions = np.concatenate((predictions, preds_numpy), axis=0)
                groundtruth = np.concatenate((groundtruth, gt_numpy), axis=0)

        if filename is not None:
            np.savez_compressed(filename, groundtruth=groundtruth, predictions=predictions)
        return {'preds': predictions, 'targets': groundtruth}

    def get_predictions_xarray(self, model: nn.Module, **kwargs) -> xr.Dataset:
        numpy_preds_targets = self.get_predictions(model, **kwargs)
        preds, targets = numpy_preds_targets['preds'], numpy_preds_targets['targets']
        var_shape = preds.shape[:-1]
        dim_names = ['snapshot', 'latitude', 'longitude'] if len(var_shape) == 3 else ['snapshot', 'spatial_dim']

        data_vars = dict()
        for i, output_var in enumerate(self.hparams.output_vars):  # usually ['tas_pre', 'psl_pre', 'pr_pre']
            output_var_pred = preds[..., i]
            output_var_target = targets[..., i]
            data_vars[f"{output_var}_preds"] = (dim_names, output_var_pred)
            data_vars[f"{output_var}_targets"] = (dim_names, output_var_target)
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
                variable_names=";".join(self.hparams.output_vars),
            ))
        return xr_dset


"""
def sunet_collate(batch):
    batchShape = batch[0].shape
    varlimit = batchShape[1] - 3  # 3 output variables: tas, psl, pr

    data_in_array = np.array([item[:, 0:varlimit] for item in batch])
    data_out_array = np.array([item[:, varlimit:] for item in batch])

    data_in = torch.Tensor(data_in_array)
    data_out = torch.Tensor(data_out_array)
    return [data_in, data_out]

"""

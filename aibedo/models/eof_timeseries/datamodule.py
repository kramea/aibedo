import math
from typing import Optional, List, Sequence, Union, Any

import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from aibedo.datamodules.abstract_datamodule import get_tensor_dataset_from_numpy
from aibedo.utilities.utils import stem_var_id, get_logger, raise_error_if_invalid_value

log = get_logger(__name__)


class AIBEDO_EOF_DataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = 'C:/Users/salvarc/data',
                 input_vars=(
                         'crelSurf_pre', 'crel_pre', 'cresSurf_pre',
                         'cres_pre', 'netTOAcs_pre', 'netSurfcs_pre'
                 ),
                 output_vars=('tas_pre', 'ps_pre', 'pr_pre'),
                 simulation: str = 'historical',
                 partition: Sequence[float] = (0.8, 0.1, 0.1),
                 time_lag: int = 0,
                 batch_size: int = 32,
                 eval_batch_size: int = 512,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 seed: int = 43,

                 ):
        super().__init__()
        self.save_hyperparameters()
        self.input_vars = [(stem_var_id(var) + '_nonorm_pcs' if var != 'lsMask' else var) for var in input_vars]
        self.output_vars = [stem_var_id(var) + '_nonorm_pcs' for var in output_vars]
        raise_error_if_invalid_value(simulation, ['historical', 'piControl'], name='simulation')

    def setup(self, stage: Optional[str] = None) -> None:
        # eof.isosph5.nonorm.CESM2.piControl.r1i1p1f1.Input.Exp8.nc
        sim = self.hparams.simulation
        ds_in = xr.open_dataset(f"{self.hparams.data_dir}/eof.isosph5.nonorm.CESM2.{sim}.r1i1p1f1.Input.Exp8.nc")
        ds_out = xr.open_dataset(f"{self.hparams.data_dir}/eof.isosph5.nonorm.CESM2.{sim}.r1i1p1f1.Output.nc")
        ml_input = self._concat_variables_into_channel_dim(ds_in, self.input_vars)
        ml_output = self._concat_variables_into_channel_dim(ds_out, self.output_vars)
        # print(ml_input.shape, ml_output.shape)
        if self.hparams.time_lag > 0:
            horizon = self.hparams.time_lag
            ml_input = ml_input[:-horizon, ...]
            ml_output = ml_output[horizon:, ...]

        train_frac, val_frac, test_frac = self.hparams.partition
        X_train, X_val, Y_train, Y_val = train_test_split(ml_input, ml_output, train_size=train_frac,
                                                          random_state=self.hparams.seed)
        X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, test_size=test_frac / (val_frac + test_frac),
                                                        random_state=self.hparams.seed)

        self._data_train = get_tensor_dataset_from_numpy(X_train, Y_train, dataset_id='train')
        self._data_val = get_tensor_dataset_from_numpy(X_val, Y_val, dataset_id='val')
        self._data_test = get_tensor_dataset_from_numpy(X_test, Y_test, dataset_id='test')

        # Data has shape (#examples, #pixels, #channels)
        if stage in ["fit", None]:
            log.info(f" Dataset sizes train: {len(self._data_train)}, val: {len(self._data_val)}")
        elif stage in ["test"]:
            log.info(f" Dataset test size: {len(self._data_test)}")
        return ml_input, ml_output

    def _concat_variables_into_channel_dim(self, data: xr.Dataset, variables: List[str], filename=None) -> np.ndarray:
        """Concatenate xarray variables into numpy channel dimension (last)."""
        data_all = []
        for var in variables:
            # Get the variable from the dataset (as numpy array, by selecting .values)
            var_data = data[var].values
            # add feature dimension (channel)
            var_data = np.expand_dims(var_data, axis=-1)
            # add to list of all variables
            data_all.append(var_data)

        # Concatenate all the variables into a single array along the last (channel/feature) dimension
        dataset = np.concatenate(data_all, axis=-1)
        assert dataset.shape[-1] == len(variables), "Number of variables does not match number of channels."
        return dataset

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




if __name__ == "__main__":
    dm = AIBEDO_EOF_DataModule()
    dm.setup()
    print(dm.ds_in.cres_nonorm_pcs.dims)
    print(dm.ds_out)

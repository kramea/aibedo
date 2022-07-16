import os.path
from typing import Optional, List, Sequence

import torch
import xarray as xr
import numpy as np
from torch.utils.data import TensorDataset
from aibedo.datamodules.abstract_datamodule import AIBEDO_DataModule
from aibedo.utilities.utils import get_logger, raise_error_if_invalid_value
from aibedo.skeleton_framework.data_loader import shuffle_data
from aibedo.skeleton_framework.spherical_unet.utils.samplings import icosahedron_nodes_calculator

log = get_logger(__name__)


class IcosahedronDatamodule(AIBEDO_DataModule):
    def __init__(self,
                 order: int = 5,
                 time_lag: int = 0,
                 time_length=None,  # TODO: deprecate this
                 partition: Sequence[float] = (0.8, 0.1, 0.1),
                 **kwargs
                 ):
        """
        Args:
            order (int): order of an icosahedron graph. Either 5 or 6.
            time_lag (int): time lag of the model.
            time_length (int): time length of the model.
            partition (tuple): partition of the data into train, validation and test fractions/sets.
                Train and validation (indices 0 and 1) must be floats.
                Test (index 2) can be a float or a string.
                    If test is a string, it must be one of the following: 'merra2', 'era5'
            kwargs: Additional keyword arguments for the super class (input_vars, data_dir, num_workers, batch_size,..).
        """
        super().__init__(**kwargs)
        # The following makes all args available as, e.g.: self.hparams.order, self.hparams.batch_size
        self.save_hyperparameters(ignore=[])
        self.n_pixels = icosahedron_nodes_calculator(self.hparams.order)
        # compress.isosph5.CESM2.historical.r1i1p1f1.Input.Exp8_fixed.nc
        self._check_args()

    @property
    def files_id(self):
        return "isosph5" if 'isosph5.' in self.hparams.input_filename else "isosph"

    def _check_args(self):
        """Check if the arguments are valid."""
        assert self.hparams.order in [5, 6], "Order of the icosahedron graph must be either 5 or 6."
        assert (self.hparams.order == 5 and 'isosph5' in self.hparams.input_filename) or self.hparams.order == 6
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

    def _concat_variables_into_channel_dim(self, data: xr.Dataset, variables: List[str], filename=None) -> np.ndarray:
        """Concatenate xarray variables into numpy channel dimension (last)."""
        data_all = []
        for var in variables:
            var_data = data[var].values
            temp_data = np.reshape(np.concatenate(var_data, axis=0), [-1, self.n_pixels, 1])
            data_all.append(temp_data)
        dataset = np.concatenate(data_all, axis=2)
        return dataset

    def _get_auxiliary_data(self, dataset_in: xr.Dataset) -> np.ndarray:
        # Pixel-wise month - assigned to each grid (every snapshot will have the same month value for each grid cell)
        months = np.arange(12)
        month_pixel_data = np.reshape(np.repeat(months, self.n_pixels), [-1, self.n_pixels, 1])  # (12, #pixels, 1)
        month_idx = [month_pixel_data for snapshot in range(165)]
        dataset_month = np.concatenate(month_idx, axis=0)  # shape (1980, #pixels, 1)

        if len(self.hparams.auxiliary_vars) > 0:
            dataset_aux = self._concat_variables_into_channel_dim(dataset_in, self.hparams.auxiliary_vars)
            dataset_aux = np.concatenate([dataset_month, dataset_aux], axis=2)  # shape (1980, #pixels, 1 + #aux-vars)
        else:
            dataset_aux = dataset_month

        return dataset_aux

    def _process_nc_dataset(self, input_filename: str, output_filename: str, shuffle: bool = False):
        input_file = os.path.join(self.hparams.data_dir, input_filename)
        output_file = os.path.join(self.hparams.data_dir, output_filename)
        in_ds = xr.open_dataset(input_file)
        out_ds = xr.open_dataset(output_file)

        in_vars, out_vars = self.hparams.input_vars, self.hparams.output_vars
        in_channels, out_channels = len(in_vars), len(out_vars)

        # Input data
        dataset_in = self._concat_variables_into_channel_dim(in_ds, in_vars, input_file)
        # Output data
        dataset_out = self._concat_variables_into_channel_dim(out_ds, out_vars, output_file)
        # Auxiliary data
        dataset_aux = self._get_auxiliary_data(in_ds)
        dataset_in = np.concatenate([dataset_in, dataset_aux], axis=2)

        dataset_in, dataset_out = self._model_specific_transform(dataset_in, dataset_out)
        if shuffle:
            dataset_in, dataset_out = shuffle_data(dataset_in, dataset_out)

        return dataset_in, dataset_out

    def _model_specific_transform(self, input_data: np.ndarray, output_data: np.ndarray):
        if "SphericalUNetLSTM" in self.model_config._target_:
            new_in_data, new_out_data = [], []
            time_length = self.model_config.time_len
            for i in range(0, len(input_data) - time_length):
                intemp = np.concatenate(input_data[i:i + time_length, :, :], axis=1)
                new_in_data.append(intemp)
                new_out_data.append(output_data[i + time_length - 1, :, :])

            input_data, output_data = np.asarray(new_in_data), np.asarray(new_out_data)
        if self.hparams.time_lag > 0:
            input_data = input_data[:-self.hparams.time_lag]
            output_data = output_data[self.hparams.time_lag:]
        # log.warning(f"No model specific transform applied for {self.model_config._target_}.")

        return input_data, output_data

    def setup(self, stage: Optional[str] = None):
        """Load data. Set internal variables: self._data_train, self._data_val, self._data_test."""
        glevel = self.hparams.order
        train_frac, val_frac, test_frac = self.hparams.partition
        if stage in ["fit", 'val', 'validation', None] or test_frac not in self._possible_test_sets:
            from sklearn.model_selection import train_test_split

        log.info(f" Grid level: {glevel}, # of pixels: {self.n_pixels}")

        if stage in ["fit", 'val', 'validation', None] or test_frac not in self._possible_test_sets:
            # compress.isosph5.SAM0-UNICON.historical.r1i1p1f1.Input.Exp8_fixed
            # E.g.:   input_file:  "<data-dir>/compress.isosph5.CESM2.historical.r1i1p1f1.Input.Exp8_fixed.nc"
            #         output_file: "<data-dir>/compress.isosph5.CESM2.historical.r1i1p1f1.Output.nc"
            fname_in = self.hparams.input_filename
            fname_out = fname_in.replace("Input.Exp8_fixed.nc", "Output.nc")
            train_data_in, train_data_out = self._process_nc_dataset(fname_in, fname_out, shuffle=True)
            X_train, X_val, Y_train, Y_val = train_test_split(train_data_in, train_data_out, train_size=train_frac,
                                                              random_state=self.hparams.seed)

        if test_frac in self._possible_test_sets:
            sphere = self.files_id
            if test_frac == 'merra2':
                #                  f"compress.{sphere}MERRA2_Input_Exp8_fixed.nc"
                test_input_fname = f"compress.{sphere}.MERRA2_Exp8_Input.2022Jul06.nc"
                test_output_fname = f"compress.{sphere}.MERRA2_Output.2022Jul06.nc"
            elif test_frac == 'era5':
                test_input_fname = f"compress.{sphere}.ERA5_Input_Exp8.nc"
                test_output_fname = f"compress.{sphere}.ERA5_Output_PrecipCon.nc"
            else:
                raise ValueError(f"Unknown test_frac: {test_frac}")
            if stage in ["test", "predict"]:
                X_test, Y_test = self._process_nc_dataset(test_input_fname, test_output_fname, shuffle=False)
        else:
            X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, test_size=test_frac / (val_frac + test_frac),
                                                            random_state=self.hparams.seed)

        if stage in ["predict", None]:
            self._data_predict = get_tensor_dataset_from_numpy(X_test, Y_test)
        if stage == 'fit' or stage is None:
            self._data_train = get_tensor_dataset_from_numpy(X_train, Y_train)
        if stage in ["fit", 'val', 'validation', None]:
            self._data_val = get_tensor_dataset_from_numpy(X_val, Y_val)
        if stage in ['test', None]:
            self._data_test = get_tensor_dataset_from_numpy(X_test, Y_test)

        # Data has shape (#examples, #pixels, #channels)
        if stage in ["fit", None]:
            log.info(f" Dataset sizes train: {len(self._data_train)}, val: {len(self._data_val)}")
        elif stage in ["test"]:
            log.info(f" Dataset test size: {len(self._data_test)}")
        elif stage == 'predict':
            log.info(f" Dataset predict size: {len(self._data_predict)}")

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


def get_tensor_dataset_from_numpy(*ndarrays) -> TensorDataset:
    tensors = [torch.from_numpy(ndarray) for ndarray in ndarrays]
    return TensorDataset(*tensors)

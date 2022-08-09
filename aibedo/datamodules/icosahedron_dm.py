import os.path
from typing import Optional, List, Sequence, Tuple

import torch
import xarray as xr
import numpy as np
from einops import rearrange
from torch import Tensor

from aibedo.constants import CLIMATE_MODELS_ALL
from aibedo.datamodules.abstract_datamodule import AIBEDO_DataModule
from aibedo.datamodules.torch_dataset import AIBEDOTensorDataset
from aibedo.utilities.utils import get_logger, raise_error_if_invalid_value, get_any_ensemble_id
from aibedo.skeleton_framework.data_loader import shuffle_data
from aibedo.skeleton_framework.spherical_unet.utils.samplings import icosahedron_nodes_calculator

log = get_logger(__name__)


class IcosahedronDatamodule(AIBEDO_DataModule):
    def __init__(self,
                 order: int = 5,
                 **kwargs
                 ):
        """
        Args:
            order (int): order of an icosahedron graph. Either 5 or 6.
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
        super()._check_args()

    def _concat_variables_into_channel_dim(self, data: xr.Dataset, variables: List[str], filename=None) -> np.ndarray:
        """Concatenate xarray variables into numpy channel dimension (last)."""
        data_all = []
        for var in variables:
            var_data = data[var].values
            temp_data = np.reshape(np.concatenate(var_data, axis=0), [-1, self.n_pixels, 1])
            data_all.append(temp_data)
        dataset = np.concatenate(data_all, axis=2)
        return dataset

    def _get_auxiliary_data(self, dataset_in: xr.Dataset, dataset_out: xr.Dataset) -> np.ndarray:
        # Add month of the snapshot (0-11) # -1 because month we want 0-indexed months
        month_of_snapshot = np.array(dataset_out.indexes['time'].month, dtype=np.float32) - 1  # now repeat the month for each grid cell/pixel
        dataset_month = np.repeat(month_of_snapshot, self.n_pixels).reshape([-1, self.n_pixels, 1])
        if len(self.hparams.auxiliary_vars) > 0:
            dataset_aux = self._concat_variables_into_channel_dim(dataset_out, self.hparams.auxiliary_vars)
            dataset_aux = np.concatenate([dataset_month, dataset_aux], axis=2)  # shape (1980, #pixels, 1 + #aux-vars)
        else:
            dataset_aux = dataset_month

        return dataset_aux

    def _process_nc_dataset(self,
                            input_filename: str,
                            output_filename: str,
                            shuffle: bool = False, stage=None):
        input_file = os.path.join(self.hparams.data_dir, input_filename)
        output_file = os.path.join(self.hparams.data_dir, output_filename)
        in_ds = xr.open_dataset(input_file)
        out_ds = xr.open_dataset(output_file)

        in_vars, out_vars = self.hparams.input_vars, self.hparams.output_vars
        in_channels, out_channels = len(in_vars), len(out_vars)

        # Input data
        dataset_in = self._concat_variables_into_channel_dim(in_ds, in_vars, input_file)
        # Output data
        if stage == 'predict':
            out_vars = [x.replace('_pre', '') for x in out_vars]
            log.info(f" Using raw output data from {os.path.basename(output_file)} -- Prediction targets: {out_vars}.")
        dataset_out = self._concat_variables_into_channel_dim(out_ds, out_vars, output_file)
        # Auxiliary data (e.g. output month, evaporation, ..)
        dataset_aux = self._get_auxiliary_data(in_ds, out_ds)

        # Reshape if using multiple timesteps for prediction
        if self.window > 1:
            window = self.window
            log.info(f" Using {window} timesteps for prediction.")
            in_tmp, out_tmp, aux_tmp = [], [], []
            for i in range(0, len(dataset_in) - window):
                # concatenate time axis into feature axis
                in_tmp += [rearrange(dataset_in[i:i + window, :, :], 't n d -> n (d t)')]
                # reselect the corresponding output/aux data (at the same time as latest time step of inputs)
                out_tmp += [dataset_out[i + window - 1, :, :]]
                # similarly, the auxiliary data should be aligned to the last month too!
                aux_tmp += [dataset_aux[i + window - 1, :, :]]

            dataset_in = np.asarray(in_tmp, dtype=np.float32)
            dataset_out = np.asarray(out_tmp, dtype=np.float32)
            dataset_aux = np.asarray(aux_tmp, dtype=np.float32)

        if self.hparams.time_lag > 0:
            horizon = self.hparams.time_lag
            # raise NotImplementedError("Time lag not implemented yet, need month for both inputs and outputs!")
            log.info(f" Model will be forecasting {self.hparams.time_lag} time steps ahead.")
            dataset_in = dataset_in[:-horizon, ...]
            dataset_out = dataset_out[horizon:, ...]
            dataset_aux = dataset_aux[horizon:, ...]

        # Concatenate the auxiliary data into the input data feature dimensions (dim=2)
        # Note, that this will not be fed into the model though -- the model inputs are cut off to only be dataset_in
        dataset_in = np.concatenate([dataset_in, dataset_aux], axis=2)
        dataset_in, dataset_out = self._model_specific_transform(dataset_in, dataset_out)
        if shuffle:
            dataset_in, dataset_out = shuffle_data(dataset_in, dataset_out)

        return dataset_in, dataset_out

    def _model_specific_transform(self, input_data: np.ndarray, output_data: np.ndarray):
        return input_data, output_data

    def _get_train_and_val_data(self, stage: str, input_filename: str) -> (Tensor, Tensor, Tensor, Tensor):
        from sklearn.model_selection import train_test_split
        # compress.isosph5.SAM0-UNICON.historical.r1i1p1f1.Input.Exp8_fixed
        # E.g.:   input_file:  "<data-dir>/compress.isosph5.CESM2.historical.r1i1p1f1.Input.Exp8_fixed.nc"
        #         output_file: "<data-dir>/compress.isosph5.CESM2.historical.r1i1p1f1.Output.nc"
        train_frac, val_frac, _ = self.hparams.partition
        fname_out = input_filename.replace("Input.Exp8_fixed.nc", "Output.PrecipCon.nc")
        train_data_in, train_data_out = self._process_nc_dataset(input_filename, fname_out, shuffle=True, stage=stage)
        X_train, X_val, Y_train, Y_val = train_test_split(train_data_in, train_data_out, train_size=train_frac,
                                                          random_state=self.hparams.seed)
        if stage == 'predict':
            del X_train, Y_train, train_data_in, train_data_out  # save some storage space
            X_train = Y_train = None

        return X_train, X_val, Y_train, Y_val

    def setup(self, stage: Optional[str] = None):
        """Load data. Set internal variables: self._data_train, self._data_val, self._data_test."""
        raise_error_if_invalid_value(stage, ['fit', 'validate', 'test', 'predict', None], 'stage')
        glevel = self.hparams.order
        input_filename = self.hparams.input_filename
        train_frac, val_frac, test_frac = self.hparams.partition
        log.info(f" Grid level: {glevel}, # of pixels: {self.n_pixels}")

        if stage in ["fit", 'validate', None] or test_frac not in self._possible_test_sets:
            X_train, X_val, Y_train, Y_val = self._get_train_and_val_data(stage, input_filename)

        if stage == 'predict' and self.prediction_data in ['val'] + CLIMATE_MODELS_ALL:
            pass
        elif test_frac in self._possible_test_sets or (
                stage == 'predict' and self.prediction_data in self._possible_test_sets
        ):
            sphere = self.files_id
            if test_frac == 'merra2' or (stage == 'predict' and self.prediction_data == 'merra2'):
                #                  f"compress.{sphere}MERRA2_Input_Exp8_fixed.nc"
                test_input_fname = f"compress.{sphere}.MERRA2_Exp8_Input.2022Jul06.nc"
                test_output_fname = f"compress.{sphere}.MERRA2_Output.2022Jul06.nc"
            elif test_frac == 'era5' or (stage == 'predict' and self.prediction_data == 'era5'):
                test_input_fname = f"compress.{sphere}.ERA5_Input_Exp8.nc"
                test_output_fname = f"compress.{sphere}.ERA5_Output_PrecipCon.nc"
            else:
                raise ValueError(f"Unknown test_frac: {test_frac}")
            if stage == "test" or stage == "predict":
                X_test, Y_test = self._process_nc_dataset(test_input_fname, test_output_fname, shuffle=False,
                                                          stage=stage)
        else:
            from sklearn.model_selection import train_test_split
            X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, test_size=test_frac / (val_frac + test_frac),
                                                            random_state=self.hparams.seed)

        if stage in ["predict"]:
            if self.prediction_data == 'same_as_test':
                X_predict, Y_predict = X_test, Y_test
            elif self.prediction_data == 'val':
                _, X_predict, _, Y_predict = self._get_train_and_val_data(stage='predict', input_filename=input_filename)
            elif self.prediction_data in CLIMATE_MODELS_ALL:
                in_file_esm = get_any_ensemble_id(self.hparams.data_dir, self.prediction_data)
                _, X_predict, _, Y_predict = self._get_train_and_val_data(stage='predict', input_filename=in_file_esm)
            else:
                raise ValueError(f"Unknown prediction_data: {self.prediction_data}")
            self._data_predict = get_tensor_dataset_from_numpy(X_predict, Y_predict, dataset_id='predict')
        if stage == 'fit' or stage is None:
            self._data_train = get_tensor_dataset_from_numpy(X_train, Y_train, dataset_id='train')
        if stage in ["fit", 'validate', None]:
            self._data_val = get_tensor_dataset_from_numpy(X_val, Y_val, dataset_id='val')
        if stage in ['test', None]:
            self._data_test = get_tensor_dataset_from_numpy(X_test, Y_test, dataset_id='test')

        # Data has shape (#examples, #pixels, #channels)
        if stage in ["fit", None]:
            log.info(f" Dataset sizes train: {len(self._data_train)}, val: {len(self._data_val)}")
        elif stage in ["test"]:
            log.info(f" Dataset test size: {len(self._data_test)}")
        elif stage == 'predict':
            log.info(f" Dataset predict size: {len(self._data_predict)}")



def get_tensor_dataset_from_numpy(*ndarrays, dataset_id="") -> AIBEDOTensorDataset:
    tensors = [torch.from_numpy(ndarray).float() for ndarray in ndarrays]
    return AIBEDOTensorDataset(*tensors, dataset_id=dataset_id)

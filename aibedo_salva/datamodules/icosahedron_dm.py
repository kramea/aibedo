import os.path
from typing import Optional, List, Sequence

from sklearn.model_selection import train_test_split
import xarray as xr
import numpy as np
from aibedo_salva.datamodules.abstract_datamodule import AIBEDO_DataModule
from aibedo_salva.utilities.utils import get_logger, raise_error_if_invalid_value
from aibedo_salva.skeleton_framework.data_loader import shuffle_data
from aibedo_salva.skeleton_framework.spherical_unet.utils.samplings import icosahedron_nodes_calculator

log = get_logger(__name__)


class IcosahedronDatamodule(AIBEDO_DataModule):
    def __init__(self,
                 order: int = 5,
                 time_length: int = 2,
                 time_lag: int = 0,
                 partition: Sequence[float] = (0.8, 0.1, 0.1),
                 **kwargs
                 ):
        """
        Args:
            order (int): order of an icosahedron graph
        """
        super().__init__(**kwargs)
        # The following makes all args available as, e.g.: self.hparams.order, self.hparams.batch_size
        self.save_hyperparameters(ignore=[])
        self.n_pixels = icosahedron_nodes_calculator(self.hparams.order)
        self._possible_test_sets = ['merra2', 'era5']
        self._check_args()

    def _check_args(self):
        """Check if the arguments are valid."""
        if len(self.hparams.partition) != 3:
            raise ValueError("partition must be a tuple of 3 values")
        test_frac = self.hparams.partition[2]
        if isinstance(test_frac, str):
            raise_error_if_invalid_value(test_frac, possible_values=self._possible_test_sets, name='partition[2]')
            if self.hparams.partition[0] + self.hparams.partition[1] != 1:
                log.warning(
                    "partition[0] + partition[1] does not sum to 1 and test_frac is a string. partition[1] will be set to 1 - partition[0].")
        elif self.hparams.partition[0] + self.hparams.partition[1] + self.hparams.partition[2] != 1:
            raise ValueError(
                f"partition must sum to 1, but it sums to {self.hparams.partition[0] + self.hparams.partition[1] + self.hparams.partition[2]}")

    def _concat_variables_into_channel_dim(self, data: xr.Dataset, variables: List[str], filename=None) -> np.ndarray:
        """Concatenate xarray variables into numpy channel dimension (last)."""
        data_all = []
        for var in variables:
            var_data = data[var].data
            temp_data = np.reshape(np.concatenate(var_data, axis=0), [-1, self.n_pixels, 1])
            data_all.append(temp_data)
        dataset = np.concatenate(data_all, axis=2)
        return dataset

    def _process_nc_dataset(self, input_filename: str, output_filename: str, shuffle: bool = False):
        input_file = os.path.join(self.hparams.data_dir, input_filename)
        output_file = os.path.join(self.hparams.data_dir, output_filename)
        inDS = xr.open_dataset(input_file)
        outDS = xr.open_dataset(output_file)

        if hasattr(inDS, "lon"):
            self.lon_list = inDS.lon.data
            self.lat_list = inDS.lat.data

        in_vars, out_vars = self.hparams.input_vars, self.hparams.output_vars
        in_channels, out_channels = len(in_vars), len(out_vars)

        # Input data
        dataset_in = self._concat_variables_into_channel_dim(inDS, in_vars, input_file)
        # Output data
        dataset_out = self._concat_variables_into_channel_dim(outDS, out_vars, output_file)

        if shuffle:
            dataset_in, dataset_out = shuffle_data(dataset_in, dataset_out)

        if self.hparams.time_lag > 0:
            dataset_in = dataset_in[:-self.hparams.time_lag]
            dataset_out = dataset_out[self.hparams.time_lag:]

        combined_data = np.concatenate((dataset_in, dataset_out), axis=2)
        return combined_data

    def setup(self, stage: Optional[str] = None):
        """Load data. Set internal variables: self._data_train, self._data_val, self._data_test."""
        glevel = self.hparams.order
        time_length = self.hparams.time_length
        train_frac, val_frac, test_frac = self.hparams.partition
        train_data = val_data = test_data = None

        log.info(f"Grid level: {glevel}, # of pixels: {self.n_pixels}, time length: {time_length}")

        if stage in ["fit", None] or test_frac not in self._possible_test_sets:
            # E.g.:   input_file:  "<data-dir>/compress.isosph5.CESM2.historical.r1i1p1f1.Input.Exp8_fixed.nc"
            #         output_file: "<data-dir>/compress.isosph5.CESM2.historical.r1i1p1f1.Output.nc"
            fname_in = self.hparams.input_filename
            fname_out = fname_in.replace("Input.Exp8_fixed.nc", "Output.nc")
            combined_train_data = self._process_nc_dataset(fname_in, fname_out, shuffle=True)
            train_data, val_data = train_test_split(combined_train_data, train_size=train_frac, random_state=self.hparams.seed)

        if test_frac in self._possible_test_sets:
            sphere = "isosph5." if 'isosph5.' in self.hparams.input_filename else "isosph."
            if test_frac == 'merra2':
                test_input_fname = f"compress.{sphere}MERRA2_Input_Exp8_fixed.nc"
                test_output_fname = f"compress.{sphere}MERRA2_Output.nc"
            elif test_frac == 'era5':
                test_input_fname = f"compress.{sphere}ERA5_Input_Exp8.nc"
                test_output_fname = f"compress.{sphere}ERA5_Output_PrecipCon.nc"
            else:
                raise ValueError(f"Unknown test_frac: {test_frac}")
            if stage in ["test", "predict"]:
                test_data = self._process_nc_dataset(test_input_fname, test_output_fname, shuffle=False)
            else:
                test_data = self._data_test  # no_op
        else:
            val_data, test_data = train_test_split(val_data, test_size=test_frac / (val_frac + test_frac),
                                                   random_state=self.hparams.seed)

        self._data_train = train_data
        self._data_val = val_data
        self._data_test = test_data
        self._data_predict = test_data

        # Data has shape (#examples, #pixels, #channels)

        if stage in ["fit", None]:
            log.info(f"Dataset sizes train: {train_data.shape[0]}, val: {val_data.shape[0]}")
        else:
            log.info(f"Dataset test size: {test_data.shape[0]}")

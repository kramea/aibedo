import logging
import os.path
from typing import Optional, List, Callable, Sequence

import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import xarray as xr
import numpy as np
from aibedo_salva.datamodules.abstract_datamodule import AIBEDO_DataModule
from aibedo_salva.utilities.utils import get_logger
from skeleton_framework.data_loader import shuffle_data
from skeleton_framework.spherical_unet.utils.samplings import icosahedron_nodes_calculator

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

    def setup(self, stage: Optional[str] = None):
        """Load data. Set internal variables: self._data_train, self._data_val, self._data_test."""
        glevel = self.hparams.order
        time_length = self.hparams.time_length

        log.info(f"Grid level: {glevel}, # of pixels: {self.n_pixels}, time length: {time_length}")
        # E.g.:   input_file:  "<data-dir>/compress.isosph5.CESM2.historical.r1i1p1f1.Input.Exp8_fixed.nc"
        #         output_file: "<data-dir>/compress.isosph5.CESM2.historical.r1i1p1f1.Output.nc"
        output_fname = self.hparams.input_filename.replace("Input.Exp8_fixed.nc", "Output.nc")
        input_file = os.path.join(self.hparams.data_dir, self.hparams.input_filename)
        output_file = os.path.join(self.hparams.data_dir, output_fname)
        inDS = xr.open_dataset(input_file)
        outDS = xr.open_dataset(output_file)

        self.lon_list = inDS.lon.data
        self.lat_list = inDS.lat.data

        in_vars, out_vars = self.hparams.input_vars, self.hparams.output_vars
        in_channels, out_channels = len(in_vars), len(out_vars)

        # Input data
        data_all = []
        for var in in_vars:
            temp_data = np.reshape(np.concatenate(inDS[var].data, axis=0), [-1, self.n_pixels, 1])
            data_all.append(temp_data)
        dataset_in = np.concatenate(data_all, axis=2)

        # Output data
        data_all = []
        for var in out_vars:
            temp_data = np.reshape(np.concatenate(outDS[var].data, axis=0), [-1, self.n_pixels, 1])
            data_all.append(temp_data)
        dataset_out = np.concatenate(data_all, axis=2)

        dataset_in, dataset_out = shuffle_data(dataset_in, dataset_out)

        if self.hparams.time_lag > 0:
            dataset_in = dataset_in[:-self.hparams.time_lag]
            dataset_out = dataset_out[self.hparams.time_lag:]

        combined_data = np.concatenate((dataset_in, dataset_out), axis=2)

        train_frac, val_frac, test_frac = self.hparams.partition
        train_data, temp = train_test_split(combined_data, train_size=train_frac, random_state=self.hparams.seed)
        val_data, test_data = train_test_split(temp, test_size=test_frac / (val_frac + test_frac),
                                               random_state=self.hparams.seed)

        self._data_train = train_data
        self._data_val = val_data
        self._data_test = test_data
        self._data_predict = test_data

        # Data has shape (#examples, #pixels, #channels)
        log.info(f"Dataset sizes train: {train_data.shape[0]}, val: {val_data.shape[0]}, test: {test_data.shape[0]}")
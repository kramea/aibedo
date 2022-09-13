import os
from os.path import join
from typing import Optional, List, Callable, Sequence, Dict, Union, Tuple

from einops import rearrange, repeat
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
from aibedo.skeleton_framework.data_loader import shuffle_data
from aibedo.utilities.constraints import AUXILIARY_VARS
from aibedo.utilities.naming import var_names_to_clean_name
from aibedo.utilities.utils import get_logger, raise_error_if_invalid_value, get_any_ensemble_id, get_input_var_to_idx, \
    get_all_present_esm_ensemble_ids

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
                 esm_for_training: Union[str, Sequence[str]] = "CESM2",
                 ensemble_ids: Union[str, List[str]] = 'any',
                 partition: Sequence[float] = (0.8, 0.1, 0.1),
                 time_lag: int = 0,
                 prediction_data: str = "same_as_test",
                 use_crel: bool = True,
                 use_crelSurf: bool = True,
                 use_cresSurf: bool = True,
                 model_config: DictConfig = None,
                 batch_size: int = 64,
                 eval_batch_size: int = 512,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 verbose: bool = True,
                 seed: int = 43,
                 input_filename: str = None,  # todo: deprecate this
                 ):
        """
        Args:
            input_vars: list of input variables/predictors, e.g. ['crelSurf_pre', 'crel_pre', 'cresSurf_pre']
            output_vars: list of output/target variables, e.g. ['tas', 'pr', 'psl']
            data_dir (str):  A path to the data folder that contains the input and output files.
            esm_for_training (str | List[str]): The name(s) of the ESM to be used for training (and validation).
            ensemble_ids (str | List[str]): The ensemble id(s) to be used for training (and validation).
                Default: 'any', i.e. the ensemble id is set "r1i1p1f1" or any other id found in the data folder.
                If 'all', all ensemble ids found in the data folder are used.
            partition (tuple): partition of the data into train, validation and test fractions/sets.
                                Train and validation (indices 0 and 1) must be floats.
                                Test (index 2) can be a float or a string.
                                   -> If test is a string, it must be one of the following: 'merra2', 'era5'
            time_lag (int): The time lag between the input and output variables (i.e. horizon of predictions).
            batch_size (int): Batch size for the training dataloader
            eval_batch_size (int): Batch size for the test and validation dataloader's
            num_workers (int): Dataloader arg for higher efficiency (usually set to # of CPU cores)
            pin_memory (bool): Dataloader arg for higher efficiency
        """
        super().__init__()
        # The following makes all args available as, e.g., self.hparams.batch_size
        self.save_hyperparameters(ignore=['model_config', 'prediction_data'])
        self.model_config = model_config
        self._spatial_dims = dict()
        self._data_train = self._data_val = self._data_test = self._data_predict = None
        self._possible_test_sets = ['merra2', 'era5']
        self._possible_prediction_sets = self._possible_test_sets + ['val', 'same_as_test'] + CLIMATE_MODELS_ALL
        self.prediction_data = prediction_data
        if isinstance(esm_for_training, str):
            self._esm_for_training = tuple(CLIMATE_MODELS_ALL) if esm_for_training == 'all' else [esm_for_training]
        else:
            self._esm_for_training = tuple(esm_for_training)
        self.hparams.auxiliary_vars = AUXILIARY_VARS if model_config.use_auxiliary_vars else []
        self.window = model_config.get("window", default_value=1)
        self.input_var_to_idx, self.input_var_names = get_input_var_to_idx(
            input_vars, self.hparams.auxiliary_vars, window=self.window
        )
        self._esm_ensemble_id = None
        self._get_normalized_targets_for_predict_too = False
        self._var_names_to_clean_name = var_names_to_clean_name()
        self._check_args()

    def masked_ensemble_input_filename(self, ESM: str) -> str:
        r""" Should be implemented by the child class.
        Make this function ensemble-agnostic by masking out the ensemble id (e.g. 'r1i1p1f1') from the filename with a *

        Returns:
            str: The filename of the input data that masks out the ensemble id with a "*".

        """
        raise NotImplementedError()

    def input_filename_for_esm(self, ESM: str, ensemble_id: str = None) -> str:
        r""" Returns the filename of the input data for the given ESM and ensemble_id.

            Args:
                ESM (str): The name of the ESM. E.g.: "CESM2"
                ensemble_id (str): The ensemble id. E.g.: "r1i1p1f1".
                    Default: ´´None´´, i.e. the ensemble id is set "r1i1p1f1" or any other id found in the data folder.

            Returns:
                str: The filename of the input data for the given ESM and ensemble_id.
        """
        masked_file = self.masked_ensemble_input_filename(ESM)
        # Get the ensemble id (e.g.'r1i1p1f1')
        esm_ensemble_id = ensemble_id or get_any_ensemble_id(self.hparams.data_dir, masked_file)
        return masked_file.replace('.*.', f'.{esm_ensemble_id}.')

    @property
    def esm_for_training(self) -> Tuple[str]:
        return self._esm_for_training

    @property
    def esm_training_data_name(self) -> str:
        if len(self.esm_for_training) == 1:
            return self.esm_for_training[0]
        elif len(self.esm_for_training) == len(CLIMATE_MODELS_ALL):
            return 'all'
        return ','.join(self.esm_for_training)

    def input_filename_to_output_filename(self, input_filename: str) -> str:
        """
        Convert an input filename to the corresponding output filename.

        Args:
            input_filename (str): input filename.

        Returns:
            str : output filename.
        """
        output_filename = input_filename.replace("Input.Exp8_fixed.nc", "Output.PrecipCon.nc")
        return output_filename

    @property
    def var_names_to_clean_name(self):
        return self._var_names_to_clean_name

    @property
    def files_id(self) -> str:
        return ""

    @property
    def spatial_dims(self) -> Dict[str, int]:
        return self._spatial_dims

    @property
    def spatial_dim_names(self) -> List[str]:
        return list(self.spatial_dims.keys())

    @spatial_dims.setter
    def spatial_dims(self, value: Dict[str, int]):
        assert isinstance(value, dict), "spatial_dim must be a dict"
        self._spatial_dims = value

    @property
    def test_set_name(self) -> str:
        train_frac, val_frac, test_frac = self.hparams.partition
        if test_frac == 'merra2':
            return 'test/MERRA2'
        elif test_frac == 'era5':
            return 'test/ERA5'
        elif isinstance(test_frac, float):
            return f'test/{self.esm_training_data_name}'
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
            return f'predict/{self.esm_training_data_name}'
        elif self.prediction_data in self._possible_test_sets or self.prediction_data in CLIMATE_MODELS_ALL:
            return f'predict/{self.prediction_data.upper()}'
        else:
            raise ValueError(f"Unknown prediction data being used: {self.prediction_data}")

    @property
    def prediction_output_variables(self) -> List[str]:
        denormalized_out_vars = [x.replace('_pre', '') for x in self.hparams.output_vars]
        if self._get_normalized_targets_for_predict_too:
            return self.hparams.output_vars + denormalized_out_vars
        else:
            return denormalized_out_vars

    def _check_args(self):
        """Check if the arguments are valid."""

        # check that the ESM is known to us
        assert len(self.esm_for_training) > 0, "esm_for_training must be a non-empty list"
        for i, esm in enumerate(self.esm_for_training):
            options = CLIMATE_MODELS_ALL + self._possible_test_sets
            raise_error_if_invalid_value(esm, options, 'esm_for_training[i]')

        # check that remove_vars are indeed not present in the input_vars
        input_vars = self.hparams.input_vars

        for use_var in ['crel', 'crelSurf', 'cresSurf']:
            var_in_input_vars = [use_var in v for v in input_vars + self.input_var_names]
            if getattr(self.hparams, f'use_{use_var}'):
                assert any(var_in_input_vars), f"{use_var} must be in input_vars if use_{use_var} is True"
            else:
                assert not any(var_in_input_vars), f"{use_var} cannot be in input_vars if use_{use_var} is False"

        # check if the train, val, test split is valid
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
        input_file = os.path.join(self.hparams.data_dir, self.input_filename_for_esm(self.esm_for_training[0]))
        inDS = xr.open_dataset(input_file)

        self.lon_list: np.ndarray = inDS.lon.values
        self.lat_list: np.ndarray = inDS.lat.values
        # lsmask_round = [(round(x) if x == x else 0.5) for x in inDS.lsMask.values[0]]
        # self.ls_mask = np.array([0 if x < 0 else 1 if x > 1 else x for x in lsmask_round])
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

    def _get_auxiliary_data(self, dataset_in: xr.Dataset, dataset_out: xr.Dataset) -> np.ndarray:
        # Add month of the snapshot (0-11)
        # subtract -1 because month we want 0-indexed months
        month_of_snapshot = np.array(dataset_out.indexes['time'].month, dtype=np.float32) - 1
        # now repeat the month for each grid cell/pixel
        dataset_month = repeat(month_of_snapshot, f"t -> t {' '.join(self.spatial_dim_names)} ()", **self.spatial_dims)
        # same as: np.repeat(month_of_snapshot, self.n_pixels).reshape([-1, self.n_pixels, 1])
        if len(self.hparams.auxiliary_vars) > 0:
            dataset_aux = self._concat_variables_into_channel_dim(dataset_out, self.hparams.auxiliary_vars)
            dataset_aux = np.concatenate([dataset_month, dataset_aux],
                                         axis=-1)  # shape (1980, spatial-dims, 1 + #aux-vars)
        else:
            dataset_aux = dataset_month

        return dataset_aux

    def _process_nc_dataset(self,
                            input_filename: str,
                            output_filename: str,
                            shuffle: bool = False, stage=None) -> (np.ndarray, np.ndarray):
        input_file = os.path.join(self.hparams.data_dir, input_filename)
        output_file = os.path.join(self.hparams.data_dir, output_filename)

        # Input data
        in_ds = xr.open_dataset(input_file)
        dataset_in = self._concat_variables_into_channel_dim(in_ds, self.hparams.input_vars, input_file)

        # Output data
        if stage == 'predict':
            out_vars = self.prediction_output_variables
            ofn = os.path.basename(output_file)
            if self._get_normalized_targets_for_predict_too:
                log.info(f" Using both raw and normalized output data from {ofn} -- Prediction targets: {out_vars}.")
            else:
                log.info(f" Using raw output data from {ofn} -- Prediction targets: {out_vars}.")
        else:
            out_vars = self.hparams.output_vars

        out_ds = xr.open_dataset(output_file)
        dataset_out = self._concat_variables_into_channel_dim(out_ds, out_vars, output_file)

        # Auxiliary data (e.g. output month, evaporation, ..)
        dataset_aux = self._get_auxiliary_data(in_ds, out_ds)

        # Reshape if using multiple timesteps for prediction
        if self.window > 1:
            window = self.window
            log.info(f" Using {window} timesteps for prediction.")
            in_tmp, out_tmp, aux_tmp = [], [], []
            for i in range(0, len(dataset_in) - window):
                # concatenate time axis into feature axis (... can stand for 1 or 2 spatial dims)
                in_tmp += [rearrange(dataset_in[i:i + window, :, :], 't ... d -> ... (d t)')]
                # reselect the corresponding output/aux data (at the same time as the latest time-step of inputs)
                out_tmp += [dataset_out[i + window - 1, :, :]]
                # similarly, the auxiliary data should be aligned to the last month too!
                aux_tmp += [dataset_aux[i + window - 1, :, :]]

            dataset_in = np.asarray(in_tmp, dtype=np.float32)
            dataset_out = np.asarray(out_tmp, dtype=np.float32)
            dataset_aux = np.asarray(aux_tmp, dtype=np.float32)

        if self.hparams.time_lag > 0:
            horizon = self.hparams.time_lag
            dataset_in = dataset_in[:-horizon, ...]
            dataset_out = dataset_out[horizon:, ...]
            dataset_aux = dataset_aux[horizon:, ...]

        # Concatenate the auxiliary data into the input data feature dimensions (dim=last)
        # Note, that this will not be fed into the model though -- the model inputs are cut off to only be dataset_in
        dataset_in = np.concatenate([dataset_in, dataset_aux], axis=-1)
        dataset_in, dataset_out = self._model_specific_transform(dataset_in, dataset_out)
        if shuffle:
            dataset_in, dataset_out = shuffle_data(dataset_in, dataset_out)

        return dataset_in, dataset_out

    def _model_specific_transform(self, input_data: np.ndarray, output_data: np.ndarray):
        return input_data, output_data

    def _get_train_and_val_data_single_xarray(self,
                                              stage: str,
                                              input_filename: str,
                                              output_filename: str = None,
                                              shuffle: bool = False
                                              ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """ Get the data corresponding to a single ESM run"""
        from sklearn.model_selection import train_test_split
        # compress.isosph5.SAM0-UNICON.historical.r1i1p1f1.Input.Exp8_fixed
        # E.g.:   input_file:  "<data-dir>/compress.isosph5.CESM2.historical.r1i1p1f1.Input.Exp8_fixed.nc"
        #         output_file: "<data-dir>/compress.isosph5.CESM2.historical.r1i1p1f1.Output.nc"
        train_frac, val_frac, _ = self.hparams.partition
        output_filename = output_filename or self.input_filename_to_output_filename(input_filename)
        train_data_in, train_data_out = self._process_nc_dataset(input_filename, output_filename,
                                                                 shuffle=shuffle, stage=stage)
        X_train, X_val, Y_train, Y_val = train_test_split(train_data_in, train_data_out, train_size=train_frac,
                                                          random_state=self.hparams.seed)
        if stage == 'predict':
            del train_data_in, train_data_out  # save some storage space
            X_train = Y_train = None

        return X_train, X_val, Y_train, Y_val

    def _get_train_and_val_data(self, stage: str) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """ Get the full data corresponding to all ESMs used for training/validating """
        if self.esm_for_training[0] in self._possible_test_sets and len(self.esm_for_training) == 1:
            # Use subset of the test set as training set
            train_set = self.esm_for_training[0]
            log.info(f" Using temporally split subset of evaluation set {train_set.upper()} as training set.")
            test_input_fname, test_output_fname = get_test_set_filenames(train_set, file_prefix=self.files_id)
            return self._get_train_and_val_data_single_xarray(
                stage, input_filename=test_input_fname, output_filename=test_output_fname, shuffle=False
            )

        if len(self.esm_for_training) > 1:
            log.info(f" Concatenating {len(self.esm_for_training)} ESM datasets for training/validation.")
        for i, esm in enumerate(self.esm_for_training):
            ensemble_id_strategy = self.hparams.ensemble_ids
            if ensemble_id_strategy == 'all':
                masked_esm_fname = self.masked_ensemble_input_filename(esm)
                ensemble_ids = get_all_present_esm_ensemble_ids(self.hparams.data_dir, masked_esm_fname)
                input_filenames = [self.input_filename_for_esm(esm, ensemble_id=ens_id) for ens_id in ensemble_ids]
            elif ensemble_id_strategy == 'any':
                input_filenames = [self.input_filename_for_esm(esm, ensemble_id=None)]
                ensemble_ids = [input_filenames[0].split('.')[-4]]
            else:
                raise ValueError(f"Unknown ensemble_id_strategy: {ensemble_id_strategy}")

            #self.trainer.logger.log_hyperparams({
            #    f"training_data/{esm}/input_filenames": input_filenames,
            #    f"training_data/{esm}/ensemble_ids": ensemble_ids,
            #})
            import wandb
            if wandb.run:
                log.info(' Updating wandb config with training data info.')
                wandb.config.update({
                    f"training_data/{esm}/input_filenames": input_filenames,
                    f"training_data/{esm}/ensemble_ids": ensemble_ids,
                })
            log.info(f" Ensembles from ESM={esm} used for training: {ensemble_ids}")

            for j, input_filename in enumerate(input_filenames):
                X_train_esm, X_val_esm, Y_train_esm, Y_val_esm = self._get_train_and_val_data_single_xarray(
                    stage, input_filename=input_filename
                )
                if stage == 'predict':
                    X_train_esm = Y_train_esm = None
                if i == 0 and j == 0:
                    X_train, X_val, Y_train, Y_val = X_train_esm, X_val_esm, Y_train_esm, Y_val_esm
                else:
                    # concatenate the data from the ESMs along the batch dimension
                    X_val = np.concatenate([X_val, X_val_esm], axis=0)
                    Y_val = np.concatenate([Y_val, Y_val_esm], axis=0)
                    if stage != 'predict':  # for predict data we don't care about the training data
                        X_train = np.concatenate([X_train, X_train_esm], axis=0)
                        Y_train = np.concatenate([Y_train, Y_train_esm], axis=0)
        return X_train, X_val, Y_train, Y_val

    def _log_at_setup_start(self, stage: Optional[str] = None):
        """Log some arguments at setup."""
        if self.hparams.time_lag > 0:
            log.info(f" Model will be forecasting {self.hparams.time_lag} time steps ahead.")

    def setup(self, stage: Optional[str] = None):
        """Load data. Set internal variables: self._data_train, self._data_val, self._data_test."""
        raise_error_if_invalid_value(stage, ['fit', 'validate', 'test', 'predict', None], 'stage')
        self._log_at_setup_start(stage)

        train_frac, val_frac, test_frac = self.hparams.partition

        if stage in ["fit", 'validate', None] or test_frac not in self._possible_test_sets:
            X_train, X_val, Y_train, Y_val = self._get_train_and_val_data(stage)

        if stage == 'predict' and self.prediction_data in ['val'] + CLIMATE_MODELS_ALL:
            pass
        elif test_frac in self._possible_test_sets or (
                stage == 'predict' and self.prediction_data in self._possible_test_sets
        ):
            if test_frac == 'merra2' or (stage == 'predict' and self.prediction_data == 'merra2'):
                test_set = 'merra2'
            elif test_frac == 'era5' or (stage == 'predict' and self.prediction_data == 'era5'):
                test_set = 'era5'
            else:
                raise ValueError(f"Unknown test_frac: {test_frac}")
            test_input_fname, test_output_fname = get_test_set_filenames(test_set, file_prefix=self.files_id)

            if stage == "test" or stage == "predict":
                X_test, Y_test = self._process_nc_dataset(test_input_fname, test_output_fname, shuffle=False,
                                                          stage=stage)
        else:
            from sklearn.model_selection import train_test_split
            X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, test_size=test_frac / (val_frac + test_frac),
                                                            random_state=self.hparams.seed)

        if stage in ["predict"]:
            train_val_esm = self.esm_training_data_name
            if self.prediction_data == 'same_as_test':
                data_predict_id = f'{train_val_esm}_test' if isinstance(test_frac, float) else test_frac.upper()
                X_predict, Y_predict = X_test, Y_test
            elif self.prediction_data == 'val':
                _, X_predict, _, Y_predict = self._get_train_and_val_data(stage='predict')
                data_predict_id = f'{train_val_esm}_val'
            elif self.prediction_data in CLIMATE_MODELS_ALL:
                predict_esm = self.prediction_data
                predict_filename = self.input_filename_for_esm(ESM=predict_esm)
                _, X_predict, _, Y_predict = self._get_train_and_val_data_single_xarray(stage='predict',
                                                                                        input_filename=predict_filename)
                data_predict_id = self.prediction_data
            else:
                options = ['same_as_test', 'val'] + CLIMATE_MODELS_ALL
                raise ValueError(f"Unknown ``prediction_data``: {self.prediction_data}. Should be one of {options}")
            self._data_predict = get_tensor_dataset_from_numpy(X_predict, Y_predict, dataset_id='predict',
                                                               name=data_predict_id)
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
        from tqdm import tqdm
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model: BaseModel = model.to(device)
        predict_loader = self.predict_dataloader() if dataloader is None else dataloader
        if predict_loader.dataset is None:
            if dataloader is not None:
                log.warning(f" The dataloader provided has no dataset. Using the default predict_dataloader instead.")
            self.setup(stage='predict')
            predict_loader = self.predict_dataloader()

        predict_dataset: AIBEDOTensorDataset = predict_loader.dataset
        if predict_dataset.dataset_id == 'predict':
            log.info(' Assuming that the used dataloader has raw/non-normalized targets.')
            RAW_TARGETS = True
        else:
            RAW_TARGETS = False

        preds, targets = dict(), dict()
        for i, batch in tqdm(enumerate(predict_loader), total=len(predict_loader)):
            data_in, data_out = batch
            # get targets/preds dict of output var name -> tensor (in denormalized scale!)
            MONTH_IDX = self.input_var_to_idx['month']
            # idx 0 is arbitrary; month_of_outputs has shape (batch_size,)
            month_of_outputs = data_in[:, 0, MONTH_IDX] if data_in.dim() == 3 else data_in[:, 0, 0, MONTH_IDX]

            prediction_kwargs['month_of_outputs'] = month_of_outputs
            batch_preds = model.predict(data_in.to(device), **prediction_kwargs)
            if RAW_TARGETS:
                # only split the targets by output variable
                batch_targets = {
                    var_name: data_out[..., i]  # index the tensor along the last dimension
                    for i, var_name in enumerate(self.prediction_output_variables)
                }
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

        return {'preds': preds, 'targets': targets, 'dataset_name': predict_dataset.name}

    def get_predictions_xarray(self,
                               model: BaseModel,
                               variables='all',
                               also_targets: bool = True,
                               also_errors: bool = False,
                               return_normalized_outputs: bool = False,
                               **prediction_kwargs
                               ) -> xr.Dataset:
        self._set_geographical_metadata()
        self._get_normalized_targets_for_predict_too = return_normalized_outputs
        numpy_preds_targets = self.get_predictions(model, **prediction_kwargs,
                                                   return_normalized_outputs=return_normalized_outputs)
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
                description=f"AiBEDO predictions.",
                dataset_name=numpy_preds_targets['dataset_name'],
                variable_names=";".join(out_vars),
            ))
        return xr_dset


def get_tensor_dataset_from_numpy(*ndarrays, dataset_id="", name='', **kwargs) -> AIBEDOTensorDataset:
    tensors = [torch.from_numpy(ndarray).float() for ndarray in ndarrays]
    return AIBEDOTensorDataset(*tensors, dataset_id=dataset_id, name=name, **kwargs)

def get_test_set_filenames(test_set_name: str, file_prefix: str) -> (str, str):
    if test_set_name == 'merra2':
        #                  f"compress.{sphere}MERRA2_Input_Exp8_fixed.nc"
        test_input_fname = f"{file_prefix}MERRA2_Exp8_Input.2022Jul06.nc"
        test_output_fname = f"{file_prefix}MERRA2_Output.2022Jul06.nc"
    elif test_set_name == 'era5':
        test_input_fname = f"{file_prefix}ERA5_Input_Exp8.nc"
        test_output_fname = f"{file_prefix}ERA5_Output_PrecipCon.nc"
    else:
        raise ValueError(f"Unknown ´´test_set_name´´ {test_set_name}")
    return test_input_fname, test_output_fname
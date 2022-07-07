import logging
from typing import Optional, List, Callable, Sequence
from omegaconf import DictConfig

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import numpy as np
from aibedo_salva.data_transforms.normalization import Normalizer
from aibedo_salva.utilities.utils import get_logger

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
                 normalizer: Optional[Normalizer] = None,
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
        self.save_hyperparameters(ignore=["normalizer", 'model_config'])
        self.normalizer = normalizer
        self.model_config = model_config
        self._data_train = self._data_val = self._data_test = self._data_predict = None
        self.lon_list = self.lat_list = None

    def _shared_dataloader_kwargs(self) -> dict:
        return dict(num_workers=int(self.hparams.num_workers), pin_memory=self.hparams.pin_memory,
                    collate_fn=sunet_collate)

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

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self._data_predict, **self._shared_eval_dataloader_kwargs())


def sunet_collate(batch):
    batchShape = batch[0].shape
    varlimit = batchShape[1] - 3  # 3 output variables: tas, psl, pr

    data_in_array = np.array([item[:, 0:varlimit] for item in batch])
    data_out_array = np.array([item[:, varlimit:] for item in batch])

    data_in = torch.Tensor(data_in_array)
    data_out = torch.Tensor(data_out_array)
    return [data_in, data_out]

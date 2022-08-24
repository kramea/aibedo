import os.path
from typing import Optional, List, Sequence, Tuple

import torch
import xarray as xr
import numpy as np
from einops import rearrange, repeat
from torch import Tensor

from aibedo.constants import CLIMATE_MODELS_ALL
from aibedo.datamodules.abstract_datamodule import AIBEDO_DataModule
from aibedo.datamodules.torch_dataset import AIBEDOTensorDataset
from aibedo.utilities.utils import get_logger, raise_error_if_invalid_value, get_any_ensemble_id
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
        self.spatial_dims = {'n_pixels': self.n_pixels}  # single dim for the spatial dimension
        self.esm_ensemble_id = get_any_ensemble_id(self.hparams.data_dir, self.hparams.esm_for_training, self.files_id)
        self._check_args()

    @property
    def files_id(self) -> str:
        order_s = f"isosph{self.hparams.order}" if self.hparams.order <= 5 else "isosph"
        return f"compress.{order_s}."

    @property
    def input_filename(self) -> str:
        # compress.isosph5.CESM2.historical.r1i1p1f1.Input.Exp8_fixed.nc
        return f"{self.files_id}{self.hparams.esm_for_training}.historical.{self.esm_ensemble_id}.Input.Exp8_fixed.nc"

    def _check_args(self):
        """Check if the arguments are valid."""
        assert self.hparams.order in [5, 6], "Order of the icosahedron graph must be either 5 or 6."
        assert (self.hparams.order == 5 and 'isosph5' in self.input_filename) or self.hparams.order == 6
        super()._check_args()

    def _log_at_setup_start(self, stage: Optional[str] = None):
        """Log some arguments at setup."""
        log.info(f"Order of the icosahedron graph: {self.hparams.order}, # of pixels: {self.n_pixels}")
        super()._log_at_setup_start()


from typing import Optional, List, Sequence, Tuple

from aibedo.datamodules.abstract_datamodule import AIBEDO_DataModule
from aibedo.utilities.utils import get_logger, raise_error_if_invalid_value
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
        if self.hparams.input_filename is not None:
            # For backward compatibility
            log.warning(f" This is using an older model: input_filename is set to {self.hparams.input_filename}")
            assert self.hparams.esm_for_training == 'CESM2', f"Only CESM2 is supported for now., but not {self.hparams.esm_for_training}"
            self._esm_for_training = [self.hparams.input_filename.split('.')[2]]
        self._check_args()

    @property
    def files_id(self) -> str:
        order_s = f"isosph{self.hparams.order}" if self.hparams.order <= 5 else "isosph"
        if self._is_denorm_nonorm:
            return f"{order_s}.denorm_nonorm."
        else:
            return f"compress.{order_s}."

    @property
    def _is_denorm_nonorm(self):
        return any(
            [('denorm' in v or 'nonorm' in v) for v in self.hparams.input_vars + self.hparams.output_vars]
        )

    def masked_ensemble_input_filename(self, ESM: str) -> str:
        # compress.isosph5.CESM2.historical.r1i1p1f1.Input.Exp8_fixed.nc
        # isosph.denorm_nonorm.CESM2.historical.r1i1p1f1.Input.Exp8.nc
        suffix = 'Exp8' if self._is_denorm_nonorm else 'Exp8_fixed'
        return f"{self.files_id}{ESM}.historical.*.Input.{suffix}.nc"


    def input_filename_to_output_filename(self, input_filename: str) -> str:
        if self._is_denorm_nonorm:
            return input_filename.replace('Input.Exp8.nc', 'Output.nc')
        else:
            return input_filename.replace('Input.Exp8_fixed.nc', "Output.PrecipCon.nc")

    def _check_args(self):
        """Check if the arguments are valid."""
        assert self.hparams.order in [5, 6], "Order of the icosahedron graph must be either 5 or 6."
        if self._is_denorm_nonorm:
            assert self.hparams.order == 5, "Denorm_nonorm is only supported for isosph5 currently"
        super()._check_args()

    def _log_at_setup_start(self, stage: Optional[str] = None):
        """Log some arguments at setup."""
        super()._log_at_setup_start()
        log.info(f" Order of the icosahedron graph: {self.hparams.order}, # of pixels: {self.n_pixels}")
        if self._is_denorm_nonorm:
            log.info(" Running on denorm_nonorm data!")

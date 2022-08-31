from aibedo.datamodules.abstract_datamodule import AIBEDO_DataModule
from aibedo.utilities.utils import get_logger, raise_error_if_invalid_value

log = get_logger(__name__)


class EuclideanDatamodule(AIBEDO_DataModule):
    def __init__(self,
                 **kwargs
                 ):
        """
        Args:
            kwargs: Additional keyword arguments for the super class (input_vars, data_dir, num_workers, batch_size,..).
        """
        super().__init__(**kwargs)
        # The following makes all args available as, e.g.: self.hparams.order, self.hparams.batch_size
        self.save_hyperparameters(ignore=[])
        self.spatial_dims = {'lat': 192, 'lon': 288}  # two dims for the spatial dimension
        self._check_args()

    @property
    def files_id(self) -> str:
        # files are of the kind 'CESM2.historical.r1i1p1f1.Input.Exp8_fixed.nc',
        #  i.e. nothing precedes the ESM model name
        return ""

    def masked_ensemble_input_filename(self, ESM: str) -> str:
        # files are of the kind 'CESM2.historical.r1i1p1f1.Input.Exp8_fixed.nc'
        return f"{ESM}.historical.*.Input.Exp8_fixed.nc"

    def _check_args(self):
        """Check if the arguments are valid."""
        super()._check_args()


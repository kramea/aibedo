from abc import ABC
from typing import Dict, Any, Tuple, Union, Optional, Callable
from collections import OrderedDict

# import einops
import numpy as np
import torch
from torch import Tensor, nn

from aibedo_salva.utilities.utils import get_logger


class AbstractTransform(ABC):
    def __init__(self):
        self.log = get_logger(__name__)

    @property
    def output_dim(self) -> Union[int, Dict[str, int]]:
        """
        Returns:
            The number of feature dimensions that the transformed data will have.
            If the transform returns an array, output_dim should be an int.
            If the transform returns a dict of str -> array, output_dim should be a dict str -> int, that
                described the number of features for each key in the transformed output.

        """
        raise NotImplementedError(f"Output dim is not implemented by {self.__class__}")

    @property
    def save_transformed_data(self) -> bool:
        """ If True and numpy arrays are stored (in fast Dataset class, as .npz),
        the data is stored in transformed form.
        """
        return True

    @property
    def spatial_output_dim(self) -> Union[int, Dict[str, int]]:
        """
        Returns:
            The number of spatial dimensions that the transformed data will have.
            If the transform returns an array, spatial_output_dim should be an int.
            If the transform returns a dict of str -> array, output_dim should be a dict str -> int, that
                described the number of features for each key in the transformed output.

        """
        raise NotImplementedError(f"spatial_output_dim is not implemented by {self.__class__}")

    def transform(self, X: Dict[str, np.ndarray]) -> Any:
        """
        How to transform dict
            X = {
                'layer': layer_array,   # shape (#layers, #layer-features)
                'levels': levels_array, # shape (#levels, #level-features)
                'globals': globals_array (#global-features,)
                }
        to the form the model will use/receive it in forward.
        Implementation will be applied (with multi-processing) in the _get_item(.) method of the dataset
            --> IMPORTANT: the arrays in X will *not* have the batch dimension!
        """
        raise NotImplementedError(f"transform is not implemented by {self.__class__}")

    def batched_transform(self, X: Dict[str, np.ndarray]) -> Any:
        """
        How to transform dict
            X = {
                'layer': layer_array,   # shape (batch-size, #layers, #layer-features)
                'levels': levels_array, # shape (batch-size, #levels, #level-features)
                'globals': globals_array (batch-size, #global-features,)
                }
        to the form the model will use/receive it in forward.
        """
        raise NotImplementedError(f"batched_transform is not implemented by {self.__class__}")


class IdentityTransform(AbstractTransform):
    @property
    def output_dim(self) -> Union[int, Dict[str, int]]:
        return self.input_dim

    @property
    def spatial_output_dim(self) -> Dict[str, int]:
        return self.spatial_input_dim

    def transform(self, X_not_batched: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return X_not_batched

    def batched_transform(self, X: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return X


class FlattenTransform(AbstractTransform):
    """ Flattens a dict with array's as values into a 1D vector (or 2D with batch dimension)"""

    @property
    def output_dim(self) -> int:
        return sum(self.input_dim.values())

    @property
    def spatial_output_dim(self) -> int:
        return 0

    def transform(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([X[key].flatten() for key in INPUT_TYPES], axis=0)
        # return np.concatenate([torch.flatten(subX) for subX in X.values()], dim=0)

    def batched_transform(self, X: Tensor) -> Tensor:
        return torch.flatten(X, start_dim=1)
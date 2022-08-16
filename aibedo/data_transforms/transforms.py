from abc import ABC
from typing import Dict, Any, Tuple, Union, Optional, Callable
from collections import OrderedDict

# import einops
import numpy as np
import torch
from torch import Tensor, nn

from aibedo.utilities.utils import get_logger


class AbstractTransform(ABC):
    def __init__(self):
        self.log = get_logger(__name__)

    def transform(self, X: np.ndarray) -> Any:
        """
        How to transform input array before use by the actual NN layers.
        # TODO: Implementation will be applied (with multi-processing) in the _get_item(.) method of the dataset
            --> IMPORTANT: the arrays in X will *not* have the batch dimension!
        """
        raise NotImplementedError(f"transform is not implemented by {self.__class__}")

    def batched_transform(self, X: Dict[str, np.ndarray]) -> Any:
        """
                How to transform input array before use by the actual NN layers.
        """
        raise NotImplementedError(f"batched_transform is not implemented by {self.__class__}")


class IdentityTransform(AbstractTransform):

    def transform(self, X_not_batched: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return X_not_batched

    def batched_transform(self, X: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return X


class FlattenTransform(AbstractTransform):
    """ Flattens a dict with array's as values into a 1D vector (or 2D with batch dimension)"""

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X.reshape(-1)
        # return np.concatenate([torch.flatten(subX) for subX in X.values()], dim=0)

    def batched_transform(self, X: Tensor) -> Tensor:
        return torch.flatten(X, start_dim=1)
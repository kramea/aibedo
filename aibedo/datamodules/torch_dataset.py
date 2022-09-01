from typing import Tuple, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset


class AIBEDOTensorDataset(Dataset[Tuple[Tensor, ...]]):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    tensors: Tuple[Tensor, ...]

    def __init__(self,
                 *tensors: Tensor,
                 dataset_id: str = '',
                 name: str = '',
                 output_vars: Sequence[str] = None,
                 ) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors
        self.dataset_id = dataset_id
        self.name = name
        self.output_vars = None if output_vars is None else tuple(output_vars)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)

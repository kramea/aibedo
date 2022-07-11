from typing import Sequence, Optional, Dict, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from aibedo_salva.models.base_model import BaseModel
from aibedo_salva.models.modules.mlp import MLP


class AIBEDO_MLP(BaseModel):
    def __init__(self,
                 hidden_dims: Sequence[int],
                 datamodule_config: DictConfig = None,
                 net_normalization: Optional[str] = None,
                 activation_function: str = 'gelu',
                 dropout: float = 0.0,
                 residual: bool = False,
                 residual_learnable_lam: bool = False,
                 output_activation_function: Optional[Union[str, bool]] = None,
                 *args, **kwargs):
        """

        Args:
            hidden_dims: The hidden dimensions of the MLP (e.g. [100, 100, 100])
            datamodule_config: The config of the datamodule (e.g. produced by hydra <config>.yaml file parsing)
            net_normalization: One of ['batch_norm', 'layer_norm', 'none']
            activation_function: The activation function of the MLP (e.g. 'gelu')
            dropout: How much dropout to use in the MLP. Default is 0.0 (no dropout)
            residual (bool): Whether to use residual connections in the MLP. Default is False
            residual_learnable_lam (bool): Whether to use residual connections with learnable lambdas
            output_activation_function (str, bool, optional): By default no output activation function is used (None).
                If a string is passed, is must be the name of the desired output activation (e.g. 'softmax')
                If True, the same activation function is used as defined by the arg `activation_function`.
        """
        super().__init__(datamodule_config=datamodule_config, *args, **kwargs)
        # The following saves all the args that are passed to the constructor to self.hparams
        #   e.g. access them with self.hparams.hidden_dims
        self.save_hyperparameters()

        self.input_dim = self.num_input_features * self.spatial_dim
        self.output_dim = self.num_output_features * self.spatial_dim

        self.example_input_array = torch.randn(1, self.input_dim)
        self.num_layers = len(hidden_dims)

        self.mlp = MLP(
            self.input_dim, hidden_dims, self.output_dim,
            net_normalization=net_normalization,
            activation_function=activation_function, dropout=dropout,
            residual=residual, residual_learnable_lam=residual_learnable_lam,
            output_normalization=False,
            output_activation_function=output_activation_function
        )

    def forward(self, X: Tensor) -> Tensor:
        flattened_X = self.input_transform.batched_transform(X)
        flattened_Y = self.mlp(flattened_X)
        Y = flattened_Y.view(X.shape[0], self.spatial_dim, self.num_output_features)
        return Y

    def train_step_initial_log_dict(self) -> Dict[str, float]:
        log_dict = dict()
        if self.hparams.residual_learnable_lam and self.hparams.residual:
            log_dict = {f'mlp/ResLam_{i}': float(lay.rlam)
                        for i, lay in enumerate(self.mlp.hidden_layers)
                        if lay.residual}
            # for i, lay in enumerate(self.mlp.hidden_layers):
            #    if lay.residual:
            #        log_dict[f'mlp/ResLam_{i}'] = lay.rlam
        return log_dict

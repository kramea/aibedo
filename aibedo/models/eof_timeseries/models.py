import math
from typing import Optional, List, Sequence, Union, Any

import torch
from einops import rearrange
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule
from timm.models.layers import trunc_normal_
from torch import Tensor, nn

from aibedo.data_transforms.transforms import FlattenTransform
from aibedo.models.afnonet import AFNO_Block, PatchEmbed1D
from aibedo.models.modules.mlp import MLP
from aibedo.utilities.utils import stem_var_id, get_logger, get_loss, get_normalization_layer, identity

log = get_logger(__name__)


class EOF_BaseModel(LightningModule):
    def __init__(self, spatial_dim: int, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.spatial_dim = spatial_dim
        self.num_input_features = 6
        self.num_output_features = 3
        self.loss = get_loss('mse', reduction='mean')

    def training_step(self, batch, batch_idx: int):
        X, Y = batch
        Y_hat = self(X)
        loss = self.loss(Y_hat, Y)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx: int):
        X, Y = batch
        Y_hat = self(X)
        loss = self.loss(Y_hat, Y)
        self.log('val/loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx: int):
        X, Y = batch
        Y_hat = self(X)
        loss = self.loss(Y_hat, Y)
        self.log('test/loss', loss, prog_bar=True)
        return {'test_loss': loss, 'targets': Y, 'preds': Y_hat}

    def test_epoch_end(self, outputs: List[Any]):
        outputs = self._evaluation_get_preds(outputs)
        return outputs

    def _evaluation_get_preds(self, outputs: List[Any]):
        targets = torch.cat([batch['targets'] for batch in outputs], dim=0).cpu().numpy()
        preds = torch.cat([batch['preds'] for batch in outputs], dim=0).detach().cpu().numpy()
        return {'targets': targets, 'preds': preds}

    def configure_optimizers(self, lr: float = 2e-4, weight_decay: float = 1e-5):
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def get_preds(self, dataloader):
        self.eval()
        outputs = []
        for batch in dataloader:
            X, Y = batch
            Y_hat = self(X)
            outputs.append({'targets': Y, 'preds': Y_hat})
        return self._evaluation_get_preds(outputs)


class AIBEDO_EOF_MLP(EOF_BaseModel):

    def __init__(self,
                 hidden_dims: Sequence[int],
                 net_normalization: Optional[str] = None,
                 activation_function: str = 'gelu',
                 dropout: float = 0.0,
                 residual: bool = False,
                 residual_learnable_lam: bool = False,
                 output_activation_function: Optional[Union[str, bool]] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The following saves all the args that are passed to the constructor to self.hparams
        #   e.g. access them with self.hparams.hidden_dims
        self.save_hyperparameters()
        self.name = 'MLP'

        self.flatten_transform = FlattenTransform()

        mlp_total_spatial_dims = self.spatial_dim
        self.output_tensor_shape = (-1, self.spatial_dim, self.num_output_features)

        self.input_dim = self.num_input_features * mlp_total_spatial_dims
        self.output_dim = self.num_output_features * mlp_total_spatial_dims

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
        r"""Forward the input through the MLP.

        Shapes:
            - Input: :math:`(B, *, C_{in})`
            - Output: :math:`(B, *, C_{out})`,

            where :math:`B` is the batch size, :math:`*` is the spatial dimension(s) of the data,
            and :math:`C_{in}` (:math:`C_{out}`) is the number of input (output) features.
        """
        flattened_X = self.flatten_transform.batched_transform(X)
        flattened_Y = self.mlp(flattened_X)
        # Reshape back into spatially structured outputs
        Y = flattened_Y.view(self.output_tensor_shape)
        return Y


class EOF_AFNONet(EOF_BaseModel):
    def __init__(self,
                 hidden_dim: Optional[int] = 384,
                 num_layers: int = 4,
                 mlp_ratio: float = 4.0,
                 uniform_drop: bool = False,
                 dropout: float = 0.0,
                 drop_path_rate: float = 0.0,
                 net_normalization: str = 'layer_norm',
                 layer_norm_eps: float = 1e-6,
                 mlp_activation_function: str = "gelu",
                 linear_head: bool = True,
                 upsampling_mode: str = 'conv',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.name = 'AFNO'
        mixer = OmegaConf.create({
            '_target_': 'aibedo.models.modules.afno1d.AFNO1D_Mixing',
            'num_blocks': 8,  # 8
            'sparsity_threshold': 0.0,  # 0.01
            'activation_function': "relu",
            'hard_thresholding_fraction': 1.0        })
        self.input_dim = self.num_input_features
        if hidden_dim is None:
            hidden_dim = self.input_dim
            self.embedder = nn.Identity()

        project_to_head = identity
        input_dim_head = hidden_dim
        self.example_input_array = torch.randn((1, self.spatial_dim, self.input_dim))
        self.embedder = PatchEmbed1D(in_chans=self.input_dim, embed_dim=hidden_dim)
        num_patches = self.spatial_dim
        self.hidden_dim = hidden_dim

        # self.num_features = self.embed_dim = hidden_dim  # num_features for consistency with other models
        # num_patches = self.patch_embed.num_patches

        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        self.pos_emb_dropout = nn.Dropout(p=dropout)

        if uniform_drop:
            dpr = [drop_path_rate for _ in range(num_layers)]  # stochastic depth decay rule
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]  # stochastic depth decay rule

        net_norm_kwargs = dict(eps=layer_norm_eps) if net_normalization.lower() == "layer_norm" else dict()
        self.afno_blocks = nn.ModuleList([
            AFNO_Block(
                dim=hidden_dim,
                filter_config=mixer,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=dpr[i],
                double_skip=False,
                layer_norm_eps=layer_norm_eps,
                net_normalization=net_normalization,
                mlp_activation_function=mlp_activation_function,
            )
            for i in range(num_layers)
        ])

        self.net_norm = get_normalization_layer(net_normalization, hidden_dim, **net_norm_kwargs)
        self.project_to_head = project_to_head
        # Classifier head
        if linear_head:
            self.head = nn.Linear(input_dim_head, self.num_output_features)
        else:
            self.head = MLP(input_dim=input_dim_head,
                            output_dim=self.num_output_features,
                            hidden_dims=[(hidden_dim + self.num_output_features) // 2],
                            dropout=dropout,
                            residual=False,
                            activation_function=mlp_activation_function,
                            net_normalization='layer_norm')

        trunc_normal_(self.positional_embedding, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'positional_embedding', 'cls_token'}

    def forward_features(self, x):
        if x.shape[1] != self.input_dim:
            # Bring channel dimension (in the input it is the last dim) to the middle
            x = rearrange(x, 'b ... d -> b d ...')
        # Shape of x for 1D: [batch-size, #input-channels, #patches]
        # Shape of x for 2D: [batch-size, #input-channels, 288, 192]

        x = self.embedder(x)
        # Shape of x: [batch-size, #patches, #hidden-channels]

        # Add positional embedding
        x = x + self.positional_embedding
        x = self.pos_emb_dropout(x)
        # Shape of x: [batch-size, #patches, #hidden-channels]

        for blk in self.afno_blocks:
            x = blk(x, spatial_dim=(12, 12))

        x = self.net_norm(x)  # shape out: (B, S, hidden-dim)
        x = self.project_to_head(x)  # shape out: (B, *spatial-dims, linear-input-dim)
        return x

    def forward(self, x):
        x = self.forward_features(x)  # shape out: (B, *spatial-dims, #output vars)
        x = self.head(x)  # shape out: (B, *spatial-dims, #output vars)
        return x


if __name__ == "__main__":
    dm = AIBEDO_EOF_DataModule()
    dm.setup()
    print(dm.ds_in.cres_nonorm_pcs.dims)
    print(dm.ds_out)

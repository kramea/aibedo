from collections import OrderedDict
from typing import Optional

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from omegaconf import DictConfig
from timm.models.layers.drop import DropPath

from timm.models.layers import to_2tuple, trunc_normal_
from aibedo_salva.models.base_model import BaseModel
from aibedo_salva.models.modules.mlp import MLP
from aibedo_salva.utilities.utils import get_normalization_layer


class AFNONet(BaseModel):
    def __init__(self,
                 input_transform,
                 mixer: DictConfig,
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
                 *args, **kwargs):
        """
        Args:
            hidden_dim (int): embedding dimension (if None,the inferred input_dim is used as hidden_dim)
            num_layers (int): depth of transformer
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            dropout (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            net_normalization: (str): normalization layer
        """
        super().__init__(input_transform=input_transform, *args, **kwargs)
        self.save_hyperparameters()
        self.input_dim = self.num_input_features
        if hidden_dim is None:
            hidden_dim = self.input_dim
            self.embedder = nn.Identity()
        else:
            self.embedder = nn.Conv1d(self.input_dim, hidden_dim, kernel_size=1, stride=1)
        self.hidden_dim = hidden_dim
        num_patches = self.spatial_dim
        self.example_input_array = torch.randn((1, num_patches, self.input_dim))

        # self.num_features = self.embed_dim = hidden_dim  # num_features for consistency with other models
        # num_patches = self.patch_embed.num_patches

        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        self.pos_emb_dropout = nn.Dropout(p=dropout)

        if uniform_drop:
            self.log_text.info(f'using uniform droppath with expect rate {drop_path_rate}')
            dpr = [drop_path_rate for _ in range(num_layers)]  # stochastic depth decay rule
        else:
            self.log_text.info(f'using linear droppath with expect rate {drop_path_rate * 0.5}')
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

        # Classifier head
        if linear_head:
            self.head = nn.Linear(hidden_dim, self.num_output_features)
        else:
            self.head = MLP(input_dim=hidden_dim,
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
            # get channel dimension to the middle
            x = x.reshape(x.shape[0], self.input_dim, -1)
        B, C, S = x.shape  # C = input_hidden_dim/ num_channels
        x = self.embedder(x)
        # get channel dimension to the right end
        x = x.reshape(B, S, self.hidden_dim)
        x = x + self.positional_embedding
        x = self.pos_emb_dropout(x)

        for blk in self.afno_blocks:
            x = blk(x)

        x = self.net_norm(x)  # (B, S, hidden-dim)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class PatchEmbed2D(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class AFNO_Block(nn.Module):
    def __init__(self,
                 dim: int,
                 filter_config: DictConfig,
                 mlp_ratio=4.,
                 dropout: float = 0.0,
                 drop_path: float = 0.0,
                 net_normalization: str = "layer_norm",
                 layer_norm_eps: float = 1e-6,
                 double_skip: bool = False,
                 mlp_activation_function: str = "gelu",
                 ):
        super().__init__()
        net_norm_kwargs = dict(eps=layer_norm_eps) if net_normalization.lower() == "layer_norm" else dict()
        self.norm1 = get_normalization_layer(net_normalization, dim, **net_norm_kwargs)
        self.norm2 = get_normalization_layer(net_normalization, dim, **net_norm_kwargs)

        self.filter = hydra.utils.instantiate(filter_config, hidden_size=dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(input_dim=dim,
                       output_dim=dim,
                       hidden_dims=[mlp_hidden_dim],
                       dropout=dropout,
                       activation_function=mlp_activation_function,
                       net_normalization='none')  # todo: make layer norm

        self.double_skip_connection = double_skip

    def forward(self, x):
        # x has shape (batch-size, spatial/patch-dim, hidden/emb/channel-dim) throughout the forward step of AFNO
        residual = x
        x = self.norm1(x)
        x = self.filter(x)  # FFT-> spatial/token mixing -> IFFT

        if self.double_skip_connection:
            x = x + residual
            residual = x

        x = self.norm2(x)
        # xx = self.mlp(x.reshape((-1, x.shape[-1]))).reshape(x.shape)
        x = self.mlp(x)  # MLP
        # assert torch.isclose(x, xx).all()
        x = self.drop_path(x)
        x = x + residual
        return x

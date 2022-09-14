from collections import OrderedDict
from typing import Optional

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from einops import rearrange
from omegaconf import DictConfig

from timm.models.layers import to_2tuple, trunc_normal_
from aibedo.models.base_model import BaseModel
from aibedo.models.modules.afno2d import reshape_2d_tokens, AFNO2D_Upsampling
from aibedo.models.modules.mlp import MLP
from aibedo.utilities.utils import get_normalization_layer, identity, raise_error_if_invalid_type


class AFNONet(BaseModel):
    def __init__(self,
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
                 upsampling_mode: str = 'conv',
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
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.input_dim = self.num_input_features
        if hidden_dim is None:
            hidden_dim = self.input_dim
            self.embedder = nn.Identity()

        project_to_head = identity
        input_dim_head = hidden_dim
        if 'afno2d' in mixer._target_.lower():
            patch_size = (16, 24)
            self.embedder = PatchEmbed2D(in_chans=self.input_dim, embed_dim=hidden_dim, patch_size=patch_size)
            num_patches = self.embedder.num_patches
            H, W = raise_error_if_invalid_type(self.spatial_dim, [tuple, list], name='spatial_dim')
            self.example_input_array = torch.randn((1, H, W, self.input_dim))
            project_to_head = AFNO2D_Upsampling(hidden_dim, scale_by=patch_size, mode=upsampling_mode)
            input_dim_head = project_to_head.out_channels
        elif 'afno1d' in mixer._target_.lower():
            self.example_input_array = torch.randn((1, self.spatial_dim, self.input_dim))
            self.embedder = PatchEmbed1D(in_chans=self.input_dim, embed_dim=hidden_dim)
            num_patches = self.spatial_dim
        else:
            raise ValueError(f"Unknown mixer {mixer._target}")
        self.hidden_dim = hidden_dim

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


class PatchEmbed1D(nn.Module):
    def __init__(self, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=1, stride=1)

    def forward(self, x):
        B, C, S = x.shape  # C = input_hidden_dim/ num_channels
        x = self.proj(x)
        # Reshape the tensor to channels-last
        x = rearrange(x, 'b c s -> b s c')  # same as: x.reshape(B, S, C)
        return x


class PatchEmbed2D(nn.Module):
    def __init__(self, img_size=(192, 288), patch_size=(16, 24), in_chans=3, embed_dim=768):
        super().__init__()
        img_size = img_size
        patch_size = to_2tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.img_size = img_size
        # num_patches: 216, img_size: (192, 288), patch_size: (16, 16)
        # num_patches: 144, img_size: (192, 288), patch_size: (16, 24)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)  # [1, 768, 12, 12]
        x = x.flatten(2).transpose(1, 2)  # [1, 144, 768]
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
        if drop_path > 0:
            from timm.models.layers.drop import DropPath
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(input_dim=dim,
                       output_dim=dim,
                       hidden_dims=[mlp_hidden_dim],
                       dropout=dropout,
                       activation_function=mlp_activation_function,
                       net_normalization='none')  # todo: make layer norm

        self.double_skip_connection = double_skip

    def forward(self, x, spatial_dim=None):
        # x has shape (batch-size, spatial/patch-dim, hidden/emb/channel-dim) throughout the forward step of AFNO
        residual = x
        x = self.norm1(x)
        # FFT-> spatial/token mixing -> IFFT
        x = self.filter(x, spatial_size=spatial_dim)

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


if __name__ == '__main__':
    emb = PatchEmbed2D()
    x = torch.randn(1, 3, 192, 288)
    print(emb(x).shape)

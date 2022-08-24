# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

import math
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from aibedo.models.modules.afno1d import AFNO1D_Mixing
from aibedo.models.modules.upsampling import UpSampler


def reshape_2d_tokens(x: torch.Tensor, spatial_size: tuple = None) -> torch.Tensor:
    B, N, C = x.shape  # batch, num_patches, channel_dim
    if spatial_size is None:
        H = W = int(math.sqrt(N))
    else:
        H, W = spatial_size
    return x.reshape(B, H, W, C)


class AFNO2D_Mixing(AFNO1D_Mixing):
    """
    Same init as AFNO1D, but with forwarding (using 2D spatial size)

    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    """

    def forward(self, x, spatial_size=None):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, N, C = x.shape  # batch, num_patches, channel_dim

        if spatial_size is None:
            H = W = int(math.sqrt(N))
        else:
            H, W = spatial_size

        x = reshape_2d_tokens(x, spatial_size)
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        '''
        bdim = int(self.block_size * self.hidden_size_factor)
        total_modes = N // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        x_real_1 = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, bdim], device=x.device)
        x_imag_1 = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, bdim], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)
        
        x_real_1[:, :, :kept_modes] = self.act(
            self.multiply(x[:, :, :kept_modes].real, self.w1[0]) -
            self.multiply(x[:, :, :kept_modes].imag, self.w1[1]) +
            self.b1[0]
        )

        x_imag_1[:, :, :kept_modes] = self.act(
            self.multiply(x[:, :, :kept_modes].imag, self.w1[0]) +
            self.multiply(x[:, :, :kept_modes].real, self.w1[1]) +
            self.b1[1]
        )

        o2_real[:, :, :kept_modes] = (
                self.multiply(x_real_1[:, :, :kept_modes], self.w2[0]) -
                self.multiply(x_imag_1[:, :, :kept_modes], self.w2[1]) +
                self.b2[0]
        )

        o2_imag[:, :, :kept_modes] = (
                self.multiply(x_real_1[:, :, :kept_modes], self.w2[1]) +
                self.multiply(x_imag_1[:, :, :kept_modes], self.w2[0]) +
                self.b2[1]
        )
        '''

        x_real_1 = self.act(self.multiply(x.real, self.w1[0]) - self.multiply(x.imag, self.w1[1]) + self.b1[0])
        x_imag_1 = self.act(self.multiply(x.real, self.w1[1]) + self.multiply(x.imag, self.w1[0]) + self.b1[1])
        o2_real = self.multiply(x_real_1, self.w2[0]) - self.multiply(x_imag_1, self.w2[1]) + self.b2[0]
        o2_imag = self.multiply(x_real_1, self.w2[1]) + self.multiply(x_imag_1, self.w2[0]) + self.b2[1]

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.reshape(B, N, C)
        x = x.type(dtype)
        return x + bias


class AFNO2D_Upsampling(nn.Module):
    def __init__(self,
                 in_channels: int,
                 scale_by=(16, 24),
                 mode: str = 'conv'):
        super().__init__()
        upsampling_modules = []
        mh, mw = scale_by
        while True:
            if mh == 1 and mw == 1:
                break

            assert mh % 2 == 0 and mw % 2 == 0
            if mh != mw:
                assert (mw % 3 == 0 and mh < mw) or (mh % 3 == 0 and mh > mw)
                scale_factor = (2, 3) if mh < mw else (3, 2)
                mode_edge = 'bilinear' if 'conv' in mode else mode
                up = UpSampler(in_channels, in_channels//2, mode=mode_edge, scale_factor=scale_factor)
            else:
                scale_factor = (2, 2)
                up = UpSampler(in_channels, in_channels//2, mode=mode, scale_factor=scale_factor)

            mh = mh // scale_factor[0]
            mw = mw // scale_factor[1]
            in_channels = in_channels // 2
            upsampling_modules += [up]

        self._scale_h, self._scale_w = scale_by
        self.out_channels = in_channels
        self.upsampling = nn.Sequential(*upsampling_modules)

    def forward(self, x, spatial_size=None):
        B, S, C = x.shape
        if spatial_size is None:
            H_in = W_in = int(math.sqrt(S))
        else:
            H_in, W_in = spatial_size

        # Reshape into a 2D (HxW) spatial tensor (with channels in dim=1)
        x = x.reshape([B, C, H_in, W_in])
        x = self.upsampling(x)
        # Reshape the upsampled tensor back to channels-last
        x = rearrange(x, 'b c h w -> b h w c')
        # same as: x.reshape([B, H_in * self._scale_h, W_in * self._scale_w, C])
        return x




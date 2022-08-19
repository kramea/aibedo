# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

import math
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from aibedo.models.modules.afno1d import AFNO1D_Mixing


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

        if spatial_size == None:
            H = W = int(math.sqrt(N))
        else:
            H, W = spatial_size

        x = x.reshape(B, H, W, C)
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        bdim = int(self.block_size * self.hidden_size_factor)
        x_real_1 = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, bdim], device=x.device)
        x_imag_1 = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, bdim], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = N // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        '''
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

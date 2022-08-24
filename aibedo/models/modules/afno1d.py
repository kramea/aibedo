# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from aibedo.utilities.utils import get_activation_function


class AFNO1D_Mixing(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    """

    def __init__(self,
                 hidden_size: int,
                 num_blocks: int = 8,
                 sparsity_threshold: float = 0.01,
                 hard_thresholding_fraction: float = 1.0,
                 activation_function: str = "relu",
                 hidden_size_factor: float = 1.0,
                 ):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.hidden_size_factor = hidden_size_factor
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.scale = 0.02
        self.act = get_activation_function(activation_function)

        # when num_blocks = 1, the shapes are (2, 1, hidden_size, hidden_size)
        # when num_blocks = 2, the shapes are (2, 2, hidden_size/2, hidden_size/2)
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def multiply(self, input, weights):
        #   (128, 50, 4, 96) * (4, 96, 96) -> (128, 50, 4, 96)
        # = (B, f, num_blocks, d) * (num_blocks, d, d) -> (B, f, num_blocks, d)
        return torch.einsum('...bd,bdk->...bk', input, weights)

    #          torch.einsum('...bi,bio->...bo', x[:, :kept_modes].real, self.w1[0]) - \

    def forward(self, x, spatial_size=None):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, S, C = x.shape  # (batch-size, spatial-dim, channel-dim) = (B, S, C), note: C = self.hidden_size
        total_modes = S // 2 + 1

        # FFT x into frequency domain
        # to have shape (B, total_modes, C), with real/imag components each of that same shape, respectively
        x = torch.fft.rfft(x, dim=1, norm="ortho")
        x = x.reshape(B, total_modes, self.num_blocks, self.block_size)
        #  recall: self.block_size = self.hidden_size // self.num_blocks
        # -> (B, total_modes, n_blocks, block-size) = (B, total_modes, n_blocks, C/n_blocks)

        x_real_1 = torch.zeros([B, total_modes, self.num_blocks, self.block_size], device=x.device)
        x_imag_1 = torch.zeros([B, total_modes, self.num_blocks, self.block_size], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        kept_modes = int(total_modes * self.hard_thresholding_fraction)  # keep all by default
        x_real_1[:, :kept_modes] = self.act(
            self.multiply(x[:, :kept_modes].real, self.w1[0]) -
            self.multiply(x[:, :kept_modes].imag, self.w1[1]) +
            self.b1[0]
        )

        x_imag_1[:, :kept_modes] = self.act(
            self.multiply(x[:, :kept_modes].imag, self.w1[0]) +
            self.multiply(x[:, :kept_modes].real, self.w1[1]) +
            self.b1[1]
        )

        o2_real[:, :kept_modes] = (
                self.multiply(x_real_1[:, :kept_modes], self.w2[0]) -
                self.multiply(x_imag_1[:, :kept_modes], self.w2[1]) +
                self.b2[0]
        )

        o2_imag[:, :kept_modes] = (
                self.multiply(x_real_1[:, :kept_modes], self.w2[1]) +
                self.multiply(x_imag_1[:, :kept_modes], self.w2[0]) +
                self.b2[1]
        )

        # print(4, o2_real.shape, o2_imag.shape)
        x = torch.stack([o2_real, o2_imag], dim=-1)
        # print(5, x.shape)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        # print(6, x.shape)
        x = x.reshape(B, total_modes, C)
        x = torch.fft.irfft(x, n=S, dim=1, norm="ortho")  # inverse FFT back into token domain
        x = x.type(dtype)
        return x + bias

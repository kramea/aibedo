"""Encoder for Spherical UNet.
"""
# pylint: disable=W0221
from torch import nn

from spherical_unet.layers.chebyshev import SphericalChebConv
from spherical_unet.models.spherical_unet.utils import SphericalChebBN, SphericalChebBNPool


class SphericalChebBN2(nn.Module):
    """Building Block made of 2 Building Blocks (convolution, batchnorm, activation).
    """

    def __init__(self, in_channels, middle_channels, out_channels, lap, kernel_size):
        """Initialization.

        Args:
            in_channels (int): initial number of channels.
            middle_channels (int): middle number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            kernel_size (int, optional): polynomial degree.
        """

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spherical_cheb_bn_1 = SphericalChebBN(in_channels, middle_channels, lap, kernel_size)
        self.spherical_cheb_bn_2 = SphericalChebBN(middle_channels, out_channels, lap, kernel_size)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.Tensor`: output [batch x vertices x channels/features]
        """
        x = self.spherical_cheb_bn_1(x)
        x = self.spherical_cheb_bn_2(x)
        return x


class SphericalChebPool(nn.Module):
    """Building Block with a pooling/unpooling and a Chebyshev Convolution.
    """

    def __init__(self, in_channels, out_channels, lap, pooling, kernel_size):
        """Initialization.

        Args:
            in_channels (int): initial number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            pooling (:obj:`torch.nn.Module`): pooling/unpooling module.
            kernel_size (int, optional): polynomial degree.
        """
        super().__init__()
        self.pooling = pooling
        self.spherical_cheb = SphericalChebConv(in_channels, out_channels, lap, kernel_size)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.Tensor`: output [batch x vertices x channels/features]
        """
        #print("before pool: ", x.size()) #torch.Size([10, 162, 512])
        x = self.pooling(x)
        #print("after pool: ", x.size()) #after pool:  torch.Size([10, 42, 512])
        x = self.spherical_cheb(x)
        #print("after spherical_cheb(x)", x.size()) #after spherical_cheb(x) torch.Size([10, 42, 512])
        return x


class Encoder(nn.Module):
    """Encoder for the Spherical UNet.
    """

    def __init__(self, pooling, laps, kernel_size, in_channels):
        """Initialization.

        Args:
            pooling (:obj:`torch.nn.Module`): pooling layer.
            laps (list): List of laplacians.
            kernel_size (int): polynomial degree.
        """
        super().__init__()
        self.pooling = pooling
        self.kernel_size = kernel_size
        self.in_channels = in_channels
    
        self.enc_l5 = SphericalChebBN2(self.in_channels, 32, 64, laps[5], self.kernel_size) #`16-->5`
        self.enc_l4 = SphericalChebBNPool(64, 128, laps[4], self.pooling, self.kernel_size)
        self.enc_l3 = SphericalChebBNPool(128, 256, laps[3], self.pooling, self.kernel_size)
        self.enc_l2 = SphericalChebBNPool(256, 512, laps[2], self.pooling, self.kernel_size)
        self.enc_l1 = SphericalChebBNPool(512, 512, laps[1], self.pooling, self.kernel_size)
        self.enc_l0 = SphericalChebPool(512, 512, laps[0], self.pooling, self.kernel_size)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]

        Returns:
            x_enc* :obj: `torch.Tensor`: output [batch x vertices x channels/features]
        """
        #print("1", x.size())           #1 torch.Size([10, 40962, 8])
        x_enc5 = self.enc_l5(x)
        #print("2", x_enc5.size())      #2 torch.Size([10, 40962, 64])
        x_enc4 = self.enc_l4(x_enc5)
        #print("3", x_enc4.size())      #3 torch.Size([10, 10242, 128])
        x_enc3 = self.enc_l3(x_enc4)
        #print("4", x_enc3.size())      #4 torch.Size([10, 2562, 256])
        x_enc2 = self.enc_l2(x_enc3)
        #print("5", x_enc2.size())      #5 torch.Size([10, 642, 512])
        x_enc1 = self.enc_l1(x_enc2)
        #print("6", x_enc1.size())      #6 torch.Size([10, 162, 512])
        x_enc0 = self.enc_l0(x_enc1)
        #print("7", x_enc0.size())      #7 torch.Size([10, 42, 512])
        return x_enc0, x_enc1, x_enc2, x_enc3, x_enc4






class EncoderTemporalConv(Encoder):
    """Encoder for the Spherical UNet temporality with convolution.
    """

    def __init__(self, pooling, laps, sequence_length, kernel_size):
        """Initialization.

        Args:
            pooling (:obj:`torch.nn.Module`): pooling layer.
            laps (list): List of laplacians.
            sequence_length (int): The number of images used per sample.
            kernel_size (int): Polynomial degree.
        """
        super().__init__(pooling, laps, kernel_size)
        self.sequence_length = sequence_length
        self.enc_l5 = SphericalChebBN2(
            self.enc_l5.in_channels * self.sequence_length,
            self.enc_l5.in_channels * self.sequence_length,
            self.enc_l5.out_channels,
            laps[5],
            self.kernel_size,
        )

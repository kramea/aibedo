import torch
from spherical_unet.models.spherical_convlstm.convlstm import *
from spherical_unet.layers.samplings.icosahedron_pool_unpool import Icosahedron
from spherical_unet.utils.laplacian_funcs import get_equiangular_laplacians, get_healpix_laplacians, get_icosahedron_laplacians
import torch.nn.functional as F
import numpy as np


class ConvLSTM_SphericalChebBN(nn.Module):
    """Building Block with a Chebyshev Convolution, Batchnormalization, and ReLu activation.
    """

    def __init__(self, in_channels, out_channels, lap, kernel_size):
        """Initialization.

        Args:
            in_channels (int): initial number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            kernel_size (int, optional): polynomial degree. Defaults to 3.
        """
        super().__init__()
        self.convlstm = ConvLSTM(in_channels, out_channels, kernel_size, lap, True, True, False)
        self.batchnorm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.tensor`): input [batch x time x vertices x channels]

        Returns:
            :obj:`torch.tensor`: output [batch x time x vertices x channels]
        """
        x = self.convlstm(x)
        b, t, c, n =x.size()
        x = torch.reshape(x, [b*t, c, n])
        x = self.batchnorm(x)
        x = F.relu(x)
        x = torch.reshape(x, [b, t, c, n ])
        return x




class ConvLSTM_SphericalChebBNPool(nn.Module):
    """Building Block with a pooling/unpooling, a calling the SphericalChebBN block.
    """

    def __init__(self, in_channels, out_channels, lap, pooling, kernel_size):
        """Initialization.

        Args:
            in_channels (int): initial number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            pooling (:obj:`torch.nn.Module`): pooling/unpooling module.
            kernel_size (int, optional): polynomial degree. Defaults to 3.
        """
        super().__init__()
        self.pooling = pooling
        self.spherical_cheb_bn = ConvLSTM_SphericalChebBN(in_channels, out_channels, lap, kernel_size)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.tensor`: output [batch x vertices x channels/features]
        """



        b, t, c, n = x.size()
        x = torch.reshape(x, (b*t, c, n))
        x = self.pooling(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = torch.reshape(x, (b, t, c, -1))


        x = self.spherical_cheb_bn(x)
        return x



class ConvLSTM_SphericalChebBN2(nn.Module):
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
        self.spherical_cheb_bn_1 = ConvLSTM(in_channels, middle_channels, kernel_size, lap, True, True, False)
        self.spherical_cheb_bn_2 = ConvLSTM(middle_channels, out_channels, kernel_size, lap, True, True, False)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x time x vertices x channels]

        Returns:
            :obj:`torch.Tensor`: output [batch x time x vertices x channels]
        """
        x = self.spherical_cheb_bn_1(x)
        x = self.spherical_cheb_bn_2(x)
        return x




class ConvLSTM_SphericalChebPool(nn.Module):
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
        self.spherical_cheb = ConvLSTM(in_channels, out_channels, kernel_size, lap, True, True, False)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x time x vertices x channels/features]

        Returns:
            :obj:`torch.Tensor`: output [batch x time x vertices x channels/features]
        """
        b, t, c, n = x.size()
        x = torch.reshape(x, (b*t, c, n))
        x = self.pooling(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = torch.reshape(x, (b, t, c, -1))
        x = self.spherical_cheb(x)
        return x



class ConvLSTM_Encoder(nn.Module):
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

        self.enc_l5 = ConvLSTM_SphericalChebBN2(self.in_channels, 32, 64, laps[5], self.kernel_size) 
        self.enc_l4 = ConvLSTM_SphericalChebBNPool(64, 128, laps[4], self.pooling, self.kernel_size)
        self.enc_l3 = ConvLSTM_SphericalChebBNPool(128, 256, laps[3], self.pooling, self.kernel_size)
        self.enc_l2 = ConvLSTM_SphericalChebBNPool(256, 512, laps[2], self.pooling, self.kernel_size)
        self.enc_l1 = ConvLSTM_SphericalChebBNPool(512, 512, laps[1], self.pooling, self.kernel_size)
        self.enc_l0 = ConvLSTM_SphericalChebPool(512, 512, laps[0], self.pooling, self.kernel_size)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x time x vertices x channels/features]

        Returns:
            x_enc* :obj: `torch.Tensor`: output [batch x time x vertices x channels/features]
        """
        #print("1", x.size())
        x_enc5 = self.enc_l5(x)
        #print("2", x_enc5.size())
        x_enc4 = self.enc_l4(x_enc5)
        #print("3", x_enc4.size())
        x_enc3 = self.enc_l3(x_enc4)
        #print("4", x_enc3.size())
        x_enc2 = self.enc_l2(x_enc3)
        #print("5", x_enc2.size())
        x_enc1 = self.enc_l1(x_enc2)
        #print("6", x_enc1.size())
        x_enc0 = self.enc_l0(x_enc1)
        #print("7", x_enc0.size())

        return x_enc0, x_enc1, x_enc2, x_enc3, x_enc4






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
        x = torch.reshape(x, [b, t, c, n])
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
            x (:obj:`torch.tensor`): input [batch x time x vertices x channels/features]

        Returns:
            :obj:`torch.tensor`: output [batch x time x vertices x channels/features]
        """
        b, t, c, n = x.size()
        x = torch.reshape(x, (b*t, c, n))
        x = self.pooling(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = torch.reshape(x, (b, t, c, -1))
        x = self.spherical_cheb_bn(x)
        return x



class ConvLSTM_SphericalChebBNPoolCheb(nn.Module):
    """Building Block calling a SphericalChebBNPool block then a SphericalCheb.
    """

    def __init__(self, in_channels, middle_channels, out_channels, lap, pooling, kernel_size):
        """Initialization.

        Args:
            in_channels (int): initial number of channels.
            middle_channels (int): middle number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            pooling (:obj:`torch.nn.Module`): pooling/unpooling module.
            kernel_size (int, optional): polynomial degree. Defaults to 3.
        """
        super().__init__()
        self.spherical_cheb_bn_pool = ConvLSTM_SphericalChebBNPool(in_channels, middle_channels, lap, pooling, kernel_size)
        self.spherical_cheb =  ConvLSTM(middle_channels, out_channels, kernel_size, lap, True, True, False)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x time x vertices x channels/features]

        Returns:
            :obj:`torch.Tensor`: output [batch x time x vertices x channels/features]
        """
        x = self.spherical_cheb_bn_pool(x)
        x = self.spherical_cheb(x)
        return x




class ConvLSTM_SphericalChebBNPoolConcat(nn.Module):
    """Building Block calling a SphericalChebBNPool Block
    then concatenating the output with another tensor
    and calling a SphericalChebBN block.
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
        self.spherical_cheb_bn_pool = ConvLSTM_SphericalChebBNPool(in_channels, out_channels, lap, pooling, kernel_size)
        self.spherical_cheb_bn = ConvLSTM_SphericalChebBN(in_channels + out_channels, out_channels, lap, kernel_size)

    def forward(self, x, concat_data):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x time x vertices x channels/features]
            concat_data (:obj:`torch.Tensor`): encoder layer output [batch x time x vertices x channels/features]

        Returns:
            :obj:`torch.Tensor`: output [batch x time x vertices x channels/features]

        """
        #print(x.size())
        x = self.spherical_cheb_bn_pool(x)
        # pylint: disable=E1101
        #print(x.size())
        x = torch.cat((x, concat_data), dim=2)
        #print(x.size())
        # pylint: enable=E1101
        x = self.spherical_cheb_bn(x)
        return x


class ConvLSTM_Decoder(nn.Module):
    """The decoder of the Spherical UNet.
    """

    def __init__(self, unpooling, laps, kernel_size, out_channels):
        """Initialization.

        Args:
            unpooling (:obj:`torch.nn.Module`): The unpooling object.
            laps (list): List of laplacians.
        """
        super().__init__()
        self.unpooling = unpooling
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.dec_l1 = ConvLSTM_SphericalChebBNPoolConcat(512, 512, laps[1], self.unpooling, self.kernel_size)
        self.dec_l2 = ConvLSTM_SphericalChebBNPoolConcat(512, 256, laps[2], self.unpooling, self.kernel_size)
        self.dec_l3 = ConvLSTM_SphericalChebBNPoolConcat(256, 128, laps[3], self.unpooling, self.kernel_size)
        self.dec_l4 = ConvLSTM_SphericalChebBNPoolConcat(128, 64, laps[4], self.unpooling, self.kernel_size)
        self.dec_l5 = ConvLSTM_SphericalChebBNPoolCheb(64, 32, self.out_channels, laps[5], self.unpooling, self.kernel_size) #3rd para:ch
        # Switch from Logits to Probabilities if evaluating model

    def forward(self, x_enc0, x_enc1, x_enc2, x_enc3, x_enc4):
        """Forward Pass.

        Args:
            x_enc* (:obj:`torch.Tensor`): input tensors.

        Returns:
            :obj:`torch.Tensor`: output after forward pass.
        """
        #print("d1", x_enc0.size(), x_enc1.size())
        x = self.dec_l1(x_enc0, x_enc1)
        #print("d1", x.size())
        #print("d2", x.size(), x_enc2.size())
        x = self.dec_l2(x, x_enc2)
        #print("d2", x.size())
        #print("d3", x.size(), x_enc3.size())
        x = self.dec_l3(x, x_enc3)
        #print("d3", x.size())
        #print("d4", x.size(), x_enc4.size())
        x = self.dec_l4(x, x_enc4)
        #print("d4", x.size())
        #print("d5", x.size())
        x = self.dec_l5(x)
        #print("d5", x.size())
        return x



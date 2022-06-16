"""Spherical Graph Convolutional Neural Network with UNet autoencoder architecture.
"""

# pylint: disable=W0221

import torch
from torch import nn
import torch.nn.functional as F
import math

#from spherical_unet.layers.samplings.equiangular_pool_unpool import Equiangular
#from spherical_unet.layers.samplings.healpix_pool_unpool import Healpix
#from spherical_unet.layers.samplings.icosahedron_pool_unpool import Icosahedron
#from spherical_unet.models.spherical_unet.decoder import Decoder
#from spherical_unet.models.spherical_unet.encoder import Encoder
from spherical_unet.utils.laplacian_funcs import get_icosahedron_laplacians
from spherical_unet.layers.chebyshev import SphericalChebConv
from spherical_unet.models.spherical_unet.utils import SphericalChebBN, SphericalChebBNPool



class SphericalUNet(nn.Module):
    """Spherical GCNN Autoencoder.
    """

    def __init__(self, pooling_class, N, depth, laplacian_type, kernel_size,in_channels, out_channels, ratio=1):
        """Initialization.

        Args:
            pooling_class (obj): One of three classes of pooling methods
            N (int): Number of pixels in the input image
            depth (int): The depth of the UNet, which is bounded by the N and the type of pooling
            kernel_size (int): chebychev polynomial degree
            ratio (float): Parameter for equiangular sampling
        """
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling_class = Icosahedron()
        self.laps = get_icosahedron_laplacians(N, depth, laplacian_type)

        self.encoder = Encoder(self.pooling_class.pooling, self.laps, self.kernel_size, self.in_channels)
        self.decoder = Decoder(self.pooling_class.unpooling, self.laps, self.kernel_size, self.out_channels)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input to be forwarded.

        Returns:
            :obj:`torch.Tensor`: output
        """
        x_encoder = self.encoder(x)
        output = self.decoder(*x_encoder)
        return output


class IcosahedronPool(nn.Module):
    """Isocahedron Pooling, consists in keeping only a subset of the original pixels (considering the ordering of an isocahedron sampling method).
    """

    def forward(self, x):
        """Forward function calculates the subset of pixels to keep based on input size and the kernel_size.

        Args:
            x (:obj:`torch.tensor`) : [batch x pixels x features]

        Returns:
            [:obj:`torch.tensor`] : [batch x pixels pooled x features]
        """
        M = x.size(1)
        order = int(math.log((M - 2) / 10) / math.log(4))
        pool_order = order - 1
        subset_pixels_keep = int(10 * math.pow(4, pool_order) + 2)
        return x[:, :subset_pixels_keep, :]



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


class SphericalChebBNPoolCheb(nn.Module):
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
        self.spherical_cheb_bn_pool = SphericalChebBNPool(in_channels, middle_channels, lap, pooling, kernel_size)
        self.spherical_cheb = SphericalChebConv(middle_channels, out_channels, lap, kernel_size)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.Tensor`: output [batch x vertices x channels/features]
        """
        x = self.spherical_cheb_bn_pool(x)
        x = self.spherical_cheb(x)
        return x


class SphericalChebBNPoolConcat(nn.Module):
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
        self.spherical_cheb_bn_pool = SphericalChebBNPool(in_channels, out_channels, lap, pooling, kernel_size)
        self.spherical_cheb_bn = SphericalChebBN(in_channels + out_channels, out_channels, lap, kernel_size)

    def forward(self, x, concat_data):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]
            concat_data (:obj:`torch.Tensor`): encoder layer output [batch x vertices x channels/features]

        Returns:
            :obj:`torch.Tensor`: output [batch x vertices x channels/features]
        """
        x = self.spherical_cheb_bn_pool(x)
        # pylint: disable=E1101
        x = torch.cat((x, concat_data), dim=2)
        # pylint: enable=E1101
        x = self.spherical_cheb_bn(x)
        return x


class Decoder(nn.Module):
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
        self.dec_l1 = SphericalChebBNPoolConcat(512, 512, laps[1], self.unpooling, self.kernel_size)
        self.dec_l2 = SphericalChebBNPoolConcat(512, 256, laps[2], self.unpooling, self.kernel_size)
        self.dec_l3 = SphericalChebBNPoolConcat(256, 128, laps[3], self.unpooling, self.kernel_size)
        self.dec_l4 = SphericalChebBNPoolConcat(128, 64, laps[4], self.unpooling, self.kernel_size)
        self.dec_l5 = SphericalChebBNPoolCheb(64, 32, self.out_channels, laps[5], self.unpooling, self.kernel_size) #3rd para:ch
        # Switch from Logits to Probabilities if evaluating model

    def forward(self, x_enc0, x_enc1, x_enc2, x_enc3, x_enc4):
        """Forward Pass.

        Args:
            x_enc* (:obj:`torch.Tensor`): input tensors.

        Returns:
            :obj:`torch.Tensor`: output after forward pass.
        """
        x = self.dec_l1(x_enc0, x_enc1)
        x = self.dec_l2(x, x_enc2)
        x = self.dec_l3(x, x_enc3)
        x = self.dec_l4(x, x_enc4)
        x = self.dec_l5(x)
        return x



class IcosahedronUnpool(nn.Module):
    """Isocahedron Unpooling, consists in adding 1 values to match the desired un pooling size
    """

    def forward(self, x):
        """Forward calculates the subset of pixels that will result from the unpooling kernel_size and then adds 1 valued pixels to match this size

        Args:
            x (:obj:`torch.tensor`) : [batch x pixels x features]

        Returns:
            [:obj:`torch.tensor`]: [batch x pixels unpooled x features]
        """
        M = x.size(1)
        order = int(math.log((M - 2) / 10) / math.log(4))
        unpool_order = order + 1
        additional_pixels = int((10 * math.pow(4, unpool_order)) + 2)
        subset_pixels_add = additional_pixels - M
        return F.pad(x, (0, 0, 0, subset_pixels_add, 0, 0), "constant", value=1)


class Icosahedron:
    """Icosahedron class, which simply groups together the corresponding pooling and unpooling.
    """

    def __init__(self):
        """Initialize icosahedron pooling and unpooling objects.
        """
        self.__pooling = IcosahedronPool()
        self.__unpooling = IcosahedronUnpool()

    @property
    def pooling(self):
        """Get pooling.
        """
        return self.__pooling

    @property
    def unpooling(self):
        """Get unpooling.
        """
        return self.__unpooling
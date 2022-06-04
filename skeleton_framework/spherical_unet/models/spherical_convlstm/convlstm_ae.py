#Working code
import torch
from spherical_unet.models.spherical_convlstm.convlstm import *
from spherical_unet.layers.samplings.icosahedron_pool_unpool import Icosahedron
from spherical_unet.models.spherical_convlstm.convlstm_decoder_ae import ConvLSTM_Decoder
from spherical_unet.models.spherical_convlstm.convlstm_encoder import ConvLSTM_Encoder
from spherical_unet.utils.laplacian_funcs import get_equiangular_laplacians, get_healpix_laplacians, get_icosahedron_laplacians
import torch.nn.functional as F
import numpy as np

class SphericalConvLSTMUnet(nn.Module):
    """Spherical GCNN Autoencoder.
    """

    def __init__(self, pooling_class, N, depth, laplacian_type, kernel_size,  in_channels, out_channels):
        """Initialization.
        Args:
            pooling_class (obj): One of three classes of pooling methods
            N (int): Number of pixels in the input image
            depth (int): The depth of the UNet, which is bounded by the N and the type of pooling
        Example:
            model = SphericalConvLSTMAutoEncoder("icosahedron", 40962, 6, "combinatorial")
        """
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        if pooling_class == "icosahedron":
            self.pooling_class = Icosahedron()
            self.laps = get_icosahedron_laplacians(N, depth, laplacian_type)
        elif pooling_class == "healpix":
            self.pooling_class = Healpix()
            self.laps = get_healpix_laplacians(N, depth, laplacian_type)
        elif pooling_class == "equiangular":
            self.pooling_class = Equiangular()
            self.laps = get_equiangular_laplacians(N, depth, self.ratio, laplacian_type)
        else:
            raise ValueError("Error: sampling method unknown. Please use icosahedron, healpix or equiangular.")


        self.encoder = ConvLSTM_Encoder(self.pooling_class.pooling, self.laps, self.kernel_size, self.in_channels)
        self.decoder = ConvLSTM_Decoder(self.pooling_class.unpooling, self.laps, self.kernel_size, self.out_channels)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input to be forwarded.

        Returns:
            :obj:`torch.Tensor`: output
        """
        x_encoder = self.encoder(x)
        output = self.decoder(*x_encoder)
        output = output[:,:,0:1,:] # tas only
        return output

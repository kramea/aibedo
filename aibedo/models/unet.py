"""
Spherical Graph Convolutional Neural Network with UNet autoencoder architecture.
"""

# pylint: disable=W0221
import torch

from aibedo.skeleton_framework.spherical_unet.layers.samplings.equiangular_pool_unpool import Equiangular
from aibedo.skeleton_framework.spherical_unet.layers.samplings.healpix_pool_unpool import Healpix
from aibedo.skeleton_framework.spherical_unet.layers.samplings.icosahedron_pool_unpool import Icosahedron
from aibedo.skeleton_framework.spherical_unet.models.spherical_unet.decoder import Decoder
from aibedo.skeleton_framework.spherical_unet.models.spherical_unet.encoder import Encoder
from aibedo.skeleton_framework.spherical_unet.utils.laplacian_funcs import get_equiangular_laplacians, \
    get_healpix_laplacians, get_icosahedron_laplacians

from aibedo.models.base_model import BaseModel


class SphericalUNet(BaseModel):
    """ Spherical GCNN Autoencoder. """

    def __init__(self,
                 pooling_class: str,
                 depth: int,
                 laplacian_type: str,
                 kernel_size: int,
                 ratio: float = 1.0,
                 **kwargs):
        """Initialization.

        Args:
            pooling_class (obj): One of three classes of pooling methods
            N (int): Number of pixels in the input image
            depth (int): The depth of the UNet, which is bounded by the N and the type of pooling
            kernel_size (int): chebychev polynomial degree
            ratio (float): Parameter for equiangular sampling
        """
        # The call to super will automatically set the #input/output channels
        super().__init__(**kwargs)
        self.save_hyperparameters()

        N = self.spatial_dim
        if pooling_class == "icosahedron":
            self.pooling_class = Icosahedron()
            self.laps = get_icosahedron_laplacians(N, depth, laplacian_type)
        elif pooling_class == "healpix":
            self.pooling_class = Healpix()
            self.laps = get_healpix_laplacians(N, depth, laplacian_type)
        elif pooling_class == "equiangular":
            self.pooling_class = Equiangular()
            self.laps = get_equiangular_laplacians(N, depth, self.hparams.ratio, laplacian_type)
        else:
            raise ValueError("Error: sampling method unknown. Please use icosahedron, healpix or equiangular.")

        shared_kwargs = dict(laps=self.laps, kernel_size=self.hparams.kernel_size)
        self.encoder = Encoder(self.pooling_class.pooling, in_channels=self.num_input_features, **shared_kwargs)
        self.decoder = Decoder(self.pooling_class.unpooling, out_channels=self.num_output_features, **shared_kwargs)
        self.example_input_array = torch.randn((1, N, self.num_input_features))

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


class SphericalUNetLSTM(SphericalUNet):
    """ Spherical GCNN Autoencoder. """
    def __init__(self, time_len: int, **kwargs):
        self.time_length = time_len
        super().__init__(**kwargs)

    @property
    def num_input_features(self) -> int:
        return self._num_input_features * self.time_length

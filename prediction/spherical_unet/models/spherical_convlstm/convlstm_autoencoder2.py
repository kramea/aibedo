##WORKS GOOD
import torch
from spherical_unet.models.spherical_convlstm.convlstm import *
from spherical_unet.layers.samplings.icosahedron_pool_unpool import Icosahedron
from spherical_unet.utils.laplacian_funcs import get_equiangular_laplacians, get_healpix_laplacians, get_icosahedron_laplacians
import torch.nn.functional as F
import numpy as np

class SphericalConvLSTMAutoEncoder(nn.Module):
    """Spherical GCNN Autoencoder.
    """

    def __init__(self, pooling_class, N, depth, laplacian_type, input_channels, output_channels):
        """Initialization.
        Args:
            pooling_class (obj): One of three classes of pooling methods
            N (int): Number of pixels in the input image
            depth (int): The depth of the UNet, which is bounded by the N and the type of pooling
        Example:
            model = SphericalConvLSTMAutoEncoder("icosahedron", 40962, 6, "combinatorial")
        """
        super().__init__()
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
        #input_dim(channels), hidden_dim, kernel_size, num_layers, lap, batch_first=False, bias=True, return_all_layers=False)
      
        self.convlstm1 = ConvLSTM(input_channels, 128, 3, self.laps[5], True, True, False)
        self.convlstm2 = ConvLSTM(128, 512, 3, self.laps[4], True, True, False)
        self.convlstm3 = ConvLSTM(512, 512, 3, self.laps[3], True, True, False)
        self.deconvlstm2 = ConvLSTM(512, 128, 3, self.laps[4], True, True, False)
        self.deconvlstm1 = ConvLSTM(128,  output_channels,3, self.laps[5], True, True, False)
        self.pooling = self.pooling_class.pooling
        self.unpooling = self.pooling_class.unpooling

    def forward(self, x):
        """Forward Pass.
        Args:
            x : input to be forwarded.
            shape: [Batch_size, Time, Channels, Nnumber_of_input]
        Returns:
            :obj:`torch.Tensor`: output
        """
        #print(x.size()) # #A tensor of size B, T, C, N 

        d1, d2, d3, n = x.size() 
        ch=d3
        x = self.convlstm1(x)
        d1, d2, d3,  n = np.shape(x.float())
        x = torch.reshape(x, [-1, n, 1])
        x = F.relu(x)
        #x = F.sigmoid(x)

        x = self.pooling(x)
        x = torch.reshape(x, [d1, d2, d3, -1])
        x = self.convlstm2(x)
        d1, d2, d3,  n = np.shape(x.float())
        x = torch.reshape(x, [-1, n, 1])
        x = F.relu(x)
        #x = F.sigmoid(x)
        
        
        x = self.pooling(x)
        x = torch.reshape(x, [d1, d2, d3, -1])
        x = self.convlstm3(x)
        d1, d2, d3,  n = np.shape(x.float())
        x = torch.reshape(x, [-1, n, 1])
        x = F.relu(x) 
        #x = F.sigmoid(x)
        
        #######################################
        x = self.unpooling(x)
        x = torch.reshape(x, [d1, d2, d3, -1])
        x = self.deconvlstm2(x)
        d1, d2, d3,  n = np.shape(x.float())
        x = torch.reshape(x, [-1, n, 1])
        x = F.relu(x)
        #x = F.sigmoid(x)
        
        
        x = self.unpooling(x)
        x = torch.reshape(x, [d1, d2, d3, -1])
        x = self.deconvlstm1(x)
        d1, d2, d3,  n = np.shape(x.float())
        x = torch.reshape(x, [-1, n, 1])
        x = F.relu(x)
        #x = F.sigmoid(x)

        x = torch.reshape(x, [d1, d2, d3, -1])
        output = x[:,-1:,:,:]
        return output

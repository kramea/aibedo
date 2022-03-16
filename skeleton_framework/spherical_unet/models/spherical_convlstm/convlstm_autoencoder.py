##WORKS GOOD
import torch
from spherical_unet.models.spherical_convlstm.convlstm import *
from spherical_unet.layers.samplings.icosahedron_pool_unpool import Icosahedron
from spherical_unet.utils.laplacian_funcs import get_equiangular_laplacians, get_healpix_laplacians, get_icosahedron_laplacians
import torch.nn.functional as F


class SphericalConvLSTMAutoEncoder(nn.Module):
    """Spherical GCNN Autoencoder.
    """

    def __init__(self, pooling_class, N, depth, laplacian_type):
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
      
        self.convlstm1 = ConvLSTM(5, 64, 3, 1, self.laps[5], True, True, False)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.convlstm2 = ConvLSTM(64, 128, 3, 1, self.laps[4], True, True, False)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.convlstm3 = ConvLSTM(128, 256, 3, 1, self.laps[3], True, True, False)
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.convlstm4 = ConvLSTM(256, 512, 3, 1, self.laps[2], True, True, False)
        self.batchnorm4 = nn.BatchNorm1d(512)
        self.convlstm5 = ConvLSTM(512, 512, 3, 1, self.laps[1], True, True, False)
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.convlstm6 = ConvLSTM(512, 512, 3, 1, self.laps[0], True, True, False)
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.deconvlstm5 = ConvLSTM(512, 512, 3, 1, self.laps[1], True, True, False)
        self.debatchnorm5 = nn.BatchNorm1d(512)
        self.deconvlstm4 = ConvLSTM(512, 256, 3, 1, self.laps[2], True, True, False)
        self.debatchnorm4 = nn.BatchNorm1d(256)
        self.deconvlstm3 = ConvLSTM(256, 128, 3, 1, self.laps[3], True, True, False)
        self.debatchnorm3 = nn.BatchNorm1d(128)
        self.deconvlstm2 = ConvLSTM(128, 64, 3, 1, self.laps[4], True, True, False)
        self.debatchnorm2 = nn.BatchNorm1d(64)
        self.deconvlstm1 = ConvLSTM(64, 3, 3, 1, self.laps[5], True, True, False)
        self.debatchnorm1 = nn.BatchNorm1d(1)
        self.pooling = self.pooling_class.pooling
        self.unpooling = self.pooling_class.unpooling
        #self.batchnorm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """Forward Pass.
        Args:
            x : input to be forwarded.
            shape: [Batch_size, Time, Channels, Nnumber_of_input]
        Returns:
            :obj:`torch.Tensor`: output
        """
        #print(x.size()) # torch.Size([2, 10, 5, 40962]) #A tensor of size B, T, C, N 

        d1, d2, d3, n = x.size()
        ch=d3
        x,_ = self.convlstm1(x)
        d1, d2, d3,  n = x[-1].size()
        x = torch.reshape(x[-1], [-1, n, 1])
        #x = torch.reshape(x[-1], [d1, d2*d3, n])
        #x = self.batchnorm1(x)
        #x = torch.reshape(x, [-1, n, 1])
        #
        x = F.relu(x)
        x = self.pooling(x)

        #print(x.size()) #torch.Size([1280, 10242, 1])

        x = torch.reshape(x, [d1, d2, d3, -1])
        x,_ = self.convlstm2(x)
        d1, d2, d3,  n = x[-1].size()
        x = torch.reshape(x[-1], [-1, n, 1])
        x = F.relu(x)
        x = self.pooling(x)
        #print(x.size()) #torch.Size([2560, 2562, 1])


        x = torch.reshape(x, [d1, d2, d3, -1])
        x,_ = self.convlstm3(x)
        d1, d2, d3,  n = x[-1].size()
        x = torch.reshape(x[-1], [-1, n, 1])
        x = F.relu(x) 
        x = self.pooling(x)
        #print(x.size()) #torch.Size([5120, 642, 1])


        x = torch.reshape(x, [d1, d2, d3, -1])
        x,_ = self.convlstm4(x)
        d1, d2, d3,  n = x[-1].size()
        x = torch.reshape(x[-1], [-1, n, 1])
        x = F.relu(x) 
        x = self.pooling(x)
        #print(x.size()) #torch.Size([10240, 162, 1])


        x = torch.reshape(x, [d1, d2, d3, -1])
        x,_ = self.convlstm5(x)
        d1, d2, d3,  n = x[-1].size()
        x = torch.reshape(x[-1], [-1, n, 1]) 
        x = F.relu(x)
        #print(x.size()) #torch.Size([10240, 162, 1])


        x = torch.reshape(x, [d1, d2, d3, -1])
        x,_ = self.deconvlstm5(x)
        d1, d2, d3,  n = x[-1].size()
        x = torch.reshape(x[-1], [-1, n, 1])
        x = F.softmax(x) 
        x = self.unpooling(x)
        #print(x.size()) #torch.Size([10240, 642, 1])


        #print(x.size())


        x = torch.reshape(x, [d1, d2, d3, -1])
        x,_ = self.deconvlstm4(x)
        d1, d2, d3,  n = x[-1].size()
        x = torch.reshape(x[-1], [-1, n, 1])
        x = F.relu(x)
        x = self.unpooling(x)
        #print(x.size()) #torch.Size([5120, 2562, 1])


        x = torch.reshape(x, [d1, d2, d3, -1])
        x,_ = self.deconvlstm3(x)
        d1, d2, d3,  n = x[-1].size()
        x = torch.reshape(x[-1], [-1, n, 1])
        x = F.relu(x)
        x = self.unpooling(x)
        #print(x.size()) #torch.Size([2560, 10242, 1])

        x = torch.reshape(x, [d1, d2, d3, -1])
        x,_ = self.deconvlstm2(x)
        d1, d2, d3,  n = x[-1].size()
        x = torch.reshape(x[-1], [-1, n, 1])
        x = F.relu(x)
        x = self.unpooling(x)
        #print(x.size()) #torch.Size([1280, 40962, 1])

        x = torch.reshape(x, [d1, d2, d3, -1])
        x,_ = self.deconvlstm1(x)
        d1, d2, d3,  n = x[-1].size()
        x = torch.reshape(x[-1], [-1, n, 1])
        x = F.relu(x)
        x = torch.reshape(x, [d1, d2, d3, -1])
        output = x
        #print(x.size()) #torch.Size([2, 10, 1, 40962])
        return output

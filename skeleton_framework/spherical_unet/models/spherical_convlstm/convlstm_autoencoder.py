import torch
from spherical_unet.models.spherical_convlstm.convlstm import *
from spherical_unet.layers.samplings.icosahedron_pool_unpool import Icosahedron
from spherical_unet.utils.laplacian_funcs import get_equiangular_laplacians, get_healpix_laplacians, get_icosahedron_laplacians


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
        self.convlstm1 = ConvLSTM(1, 64, 3, 1, self.laps[5], True, True, True)
        self.convlstm2 = ConvLSTM(64, 128, 3, 1, self.laps[4], True, True, True)
        self.convlstm3 = ConvLSTM(128, 256, 3, 1, self.laps[3], True, True, True)
        self.convlstm4 = ConvLSTM(256, 512, 3, 1, self.laps[2], True, True, True)
        self.convlstm5 = ConvLSTM(512, 512, 3, 1, self.laps[1], True, True, True)
        self.convlstm6 = ConvLSTM(512, 512, 3, 1, self.laps[0], True, True, True)
        self.deconvlstm5 = ConvLSTM(512, 512, 3, 1, self.laps[1], True, True, True)
        self.deconvlstm4 = ConvLSTM(512, 256, 3, 1, self.laps[2], True, True, True)
        self.deconvlstm3 = ConvLSTM(256, 128, 3, 1, self.laps[3], True, True, True)
        self.deconvlstm2 = ConvLSTM(128, 64, 3, 1, self.laps[4], True, True, True)
        self.deconvlstm1 = ConvLSTM(64, 1, 3, 1, self.laps[5], True, True, True)
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
        #print(x.size())

        d1, d2, d3, n = x.size()
        ch=d3
        x,_ = self.convlstm1(x)
        d1, d2, d3,  n = x[-1].size()
        x = torch.reshape(x[-1], [-1, n, 1])
        x = self.pooling(x)

        #print(x.size())

        x = torch.reshape(x, [d1, d2, d3, -1])
        x,_ = self.convlstm2(x)
        d1, d2, d3,  n = x[-1].size()
        x = torch.reshape(x[-1], [-1, n, 1])
        x = self.pooling(x)

        #print(x.size())


        x = torch.reshape(x, [d1, d2, d3, -1])
        x,_ = self.convlstm3(x)
        d1, d2, d3,  n = x[-1].size()
        x = torch.reshape(x[-1], [-1, n, 1]) 
        x = self.pooling(x)

        #print(x.size())


        x = torch.reshape(x, [d1, d2, d3, -1])
        x,_ = self.convlstm4(x)
        d1, d2, d3,  n = x[-1].size()
        x = torch.reshape(x[-1], [-1, n, 1]) 
        x = self.pooling(x)

        #print(x.size())


        x = torch.reshape(x, [d1, d2, d3, -1])
        x,_ = self.convlstm5(x)
        d1, d2, d3,  n = x[-1].size()
        x = torch.reshape(x[-1], [-1, n, 1]) 

        #print(x.size())


        x = torch.reshape(x, [d1, d2, d3, -1])
        x,_ = self.deconvlstm5(x)
        d1, d2, d3,  n = x[-1].size()
        x = torch.reshape(x[-1], [-1, n, 1]) 
        x = self.unpooling(x)


        #print(x.size())


        x = torch.reshape(x, [d1, d2, d3, -1])
        x,_ = self.deconvlstm4(x)
        d1, d2, d3,  n = x[-1].size()
        x = torch.reshape(x[-1], [-1, n, 1])
        x = self.unpooling(x)

        #print(x.size())


        x = torch.reshape(x, [d1, d2, d3, -1])
        x,_ = self.deconvlstm3(x)
        d1, d2, d3,  n = x[-1].size()
        x = torch.reshape(x[-1], [-1, n, 1])
        x = self.unpooling(x)

        #print(x.size())

        x = torch.reshape(x, [d1, d2, d3, -1])
        x,_ = self.deconvlstm2(x)
        d1, d2, d3,  n = x[-1].size()
        x = torch.reshape(x[-1], [-1, n, 1])
        x = self.unpooling(x)

        #print(x.size())

        x = torch.reshape(x, [d1, d2, d3, -1])
        x,_ = self.deconvlstm1(x)
        #print(x[-1].size())
        x = x[-1]
        output = x



        return output

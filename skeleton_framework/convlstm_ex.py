import torch
from spherical_unet.models.spherical_convlstm.convlstm import *
from spherical_unet.models.spherical_convlstm.convlstm_autoencoder import *
from spherical_unet.layers.samplings.icosahedron_pool_unpool import Icosahedron
from spherical_unet.utils.laplacian_funcs import get_equiangular_laplacians, get_healpix_laplacians, get_icosahedron_laplacians
from spherical_unet.layers.chebyshev import SphericalChebConv


#level-0(2**0=1): 12 vertices
#level-1(2**1=2): 42 vertices
#level-2(2**2=4): 162 vertices
#level-3(2**3=8): 642 vertices
#level-4(2**4=16): 2562 vertices
#level-5(2**5=32): 10242 vertices
#level-6(2**6=64): 40962 vertices
#level-7(2**7=128): 163842 vertices
#level-8(2**8=256): 655362 vertices
#level-9(2**9=512): 2621442 vertices
n=10242

model = SphericalConvLSTMAutoEncoder("icosahedron", n, 6, "combinatorial")

x = torch.rand((16, 5, 1, n)) 

model(x)



#input_dim(channels), hidden_dim, kernel_size, num_layers, lap, batch_first=False, bias=True, return_all_layers=False):
#convlstm = ConvLSTM(2, 64, 3, 2, lap[5], True, True, True)
#for i in range(6): 
#    print(lap[i].size())
#x = torch.rand((16, 10, 2, 10242)) # B, T, C, N
#o, h = convlstm(x)

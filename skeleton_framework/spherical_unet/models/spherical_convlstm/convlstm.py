import torch.nn as nn
import torch
import os
#torch.set_default_tensor_type(torch.cuda.FloatTensor)
from spherical_unet.layers.chebyshev import SphericalChebConv
from spherical_unet.models.spherical_unet.utils import SphericalChebBN, SphericalChebBNPool

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
#referred: https://github.com/KimUyen/ConvLSTM-Pytorch/blob/master/convlstm.py
#https://discuss.pytorch.org/t/dynamic-parameter-declaration-in-forward-function/427/3

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, lap, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden_dim.
        kernel_size: int
            Size of the convolutional kernel.
        lap:(:obj:`torch.sparse.FloatTensor`): 
            laplacian
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lap = lap
        self.kernel_size = kernel_size
        self.bias = bias
        self.out_channels = 4 * self.hidden_dim
        
        self.conv = SphericalChebConv(in_channels=self.input_dim + self.hidden_dim, 
                                      out_channels = 4 * self.hidden_dim,
                                      lap=self.lap,
                                      kernel_size=self.kernel_size)

        # Initialize weights for Hadamard Products
        # found W blow up to nan. added init.normal_ here
        self.register_parameter('W_ci', None)
        self.register_parameter('W_cf', None)
        self.register_parameter('W_co', None)
        
    def reset_weight(self, size, device=''):
        self.W_ci = nn.Parameter(torch.rand(size[0], size[1], device=device))
        self.W_co = nn.Parameter(torch.rand(size[0], size[1], device=device))
        self.W_cf = nn.Parameter(torch.rand(size[0], size[1], device=device))

        #self.W_ci = nn.Parameter(torch.Tensor(size[0], size[1], device=device))
        #self.W_co = nn.Parameter(torch.Tensor(size[0], size[1], device=device))
        #self.W_cf = nn.Parameter(torch.Tensor(size[0], size[1], device=device))
        
        #self.W_ci = nn.Parameter(nn.init.normal_(torch.empty(size[0], size[1], device=device)))
        #self.W_co = nn.Parameter(nn.init.normal_(torch.empty(size[0], size[1], device=device)))
        #self.W_cf = nn.Parameter(nn.init.normal_(torch.empty(size[0], size[1], device=device)))

    def forward(self, input_tensor, cur_state):
        device = self.conv.chebconv.weight.device
        if self.W_ci is None:
            self.reset_weight((self.hidden_dim, len(self.lap)), device=device)

        input_tensor = input_tensor.to(device)
        
        h_cur, c_cur = cur_state
        h_cur = h_cur.to(device)
        c_cur = c_cur.to(device)
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined = combined.permute((0, 2, 1))
        combined_conv = self.conv(combined)
        combined_conv = combined_conv.permute((0, 2, 1)) #put back

        i_conv, f_conv, C_conv, o_conv = torch.split(combined_conv, self.hidden_dim, dim=1)
       
        
        input_gate = torch.sigmoid(i_conv + self.W_ci*c_cur )
        forget_gate = torch.sigmoid(f_conv + self.W_cf*c_cur)
        
        # Current Cell output
        C = forget_gate* c_cur + input_gate * torch.tanh(C_conv)
        output_gate = torch.sigmoid(o_conv + self.W_co*C)

        # Current Hidden State
        H = output_gate * torch.tanh(C)
        return H, C


    def init_hidden(self, batch_size, image_size):
        N = image_size
        return (torch.zeros(batch_size, self.hidden_dim, N, device=self.conv.chebconv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, N, device=self.conv.chebconv.weight.device))


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, N or T, B, C, N
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3,  True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, in_channels, out_channels,  kernel_size,  lap, batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()
        self.out_channels = out_channels
        self.batch_first = batch_first
        # We will unroll this over time steps
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels, kernel_size, lap, bias)
       
    def forward(self, X, hidden_state=None):
        """

        Parameters
        ----------
        X: todo
            4-D Tensor either of shape (t, b, c, N) or (b, t, c, N)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, N) -> (b, t, c, N)
            input_tensor = X.permute(1, 0, 2, 3)

        batch_size, seq_len, _, N = X.size() #b,t,c,n
        device = X.device
        # Initialize output
        output = torch.zeros(batch_size, seq_len, self.out_channels, N, device=device)
        # Initialize Hidden State
        H = torch.zeros(batch_size, self.out_channels, N, device=device)
        # Initialize Cell Input
        C = torch.zeros(batch_size, self.out_channels, N, device=device)
        # Unroll over time steps
        for time_step in range(seq_len):
            H, C = self.convLSTMcell(input_tensor=X[:, time_step, :, :], cur_state=[H, C])
            output[:,time_step, :] = H
        return output #size [batch, seq_len, output_dims, N]


    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states


    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

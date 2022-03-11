import torch
from spherical_unet.models.spherical_convlstm.convlstm_vanilla import *
#from convlstm_vanilla import *
import torch.nn.functional as F


# Run example
#out = model(torch.randn((16, 10, 5, 192, 288)))# Batch,Time,Channels,NumberofData, output: last_state_list(h), layer_output (o)
#print(type(out), len(out), len(out[0]), len(out[1]))  #<class 'tuple'> 2 1 1
#print(len(out[0][0]),len(out[1][0])) #16, 2
#print(np.shape(out[1][0][0].detach().numpy())) # layer_output1 (16, 32, 192, 288)
#print(np.shape(out[1][0][1].detach().numpy())) # layer_output2 (16, 32, 192, 288)
#print(np.shape(out[0][0].detach().numpy())) # (16, 10, 32, 192, 288)
#
class ConvLSTM_model(nn.Module):
    """Many to one ConvLSTM.
    """

    def __init__(self, channels, w, d):
        """Initialization.
        Args:
            channels, w, d (int)
        Example:
            model = ConvLSTM_model(5, 192, 288 )
       """
        super().__init__()

        self.convlstm1 =  ConvLSTM(channels, 16, (3,3), 1, True, True, False )
        self.convlstm2 =  ConvLSTM(16, 16, (3,3), 1, True, True, False )
        self.convlstm3 =  ConvLSTM(16, 24, (3,3), 1, True, True, False )
        self.convlstm4 =  ConvLSTM(24, 1, (3,3), 1, True, True, False )

    def forward(self, x):
        """Forward Pass.
        Args:
            x : input to be forwarded.
            shape: [Batch_size, Time, Channels, Width, Depth]
        Returns:
            :obj:`torch.Tensor`: output
        """
        #print(x.size()) #torch.Size([1, 10, 5, 192, 288]) 
        x, _ = self.convlstm1(x)
        x = x[0]
        x = F.relu(x)
        #print(x.size()) #torch.Size([1, 10, 32, 192, 288])
        x, _ = self.convlstm2(x)
        x = x[0]
        x = F.relu(x)
        #print(x.size()) #torch.Size([1, 10, 64, 192, 288])
        x, _ = self.convlstm3(x)
        x = x[0]
        x = F.relu(x)
        #print(x.size()) #torch.Size([1, 10, 1, 192, 288])
        x, _ = self.convlstm4(x)
        x = x[0]
        x = F.softmax(x)
        #print(x.size()) #torch.Size([1, 10, 1, 192, 288])
        #x = x[:, -1:, :, :,:]

        return x



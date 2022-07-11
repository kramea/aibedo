import torch
import torch.nn as nn
import torch.nn.functional as F

# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 5), 5x5 kernels
        self.conv1 = nn.Conv2d(1, 5, 3, padding=1)  
        # conv layer (depth from 5 --> 10), 5x5 kernels
        self.conv2 = nn.Conv2d(5, 10, 3, padding=1)
        # conv layer (depth from 10 --> 20), 5x5 kernels
        self.conv3 = nn.Conv2d(10, 20, 3, padding=1)
        # conv layer (depth from 20 --> 30), 5x5 kernels
        self.conv4 = nn.Conv2d(20, 30 , 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(30, 20, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(20, 10, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(10, 5, 2, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(5, 1, 2, stride=2)


    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        #print(x.size())
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        #print(x.size())
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        #print(x.size())
        # add third hidden layer
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        #print(x.size())
        # add forth hidden layer
        x = F.relu(self.conv4(x))
        x = self.pool(x)  # compressed representation   
        #print(x.size())
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        #print(x.size())
        x = F.relu(self.t_conv2(x))
        #print(x.size())
        x = F.relu(self.t_conv3(x))
        #print(x.size())
        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.relu(self.t_conv4(x))
        #print(x.size())
        return x

# initialize the NN
#model = ConvAutoencoder()
#model(torch.randn(1,1,192,288))
#print(model)

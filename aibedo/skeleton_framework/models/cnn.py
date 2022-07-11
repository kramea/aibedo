import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    # padding = 2 makes output same size with input when kernel = 5
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=1, padding=2)
    self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=5, stride=1, padding=2)
    self.conv3 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=5, stride=1, padding=2)
    self.dropout =  nn.Dropout(p=0.2)
    self.conv4 = nn.Conv2d(in_channels=10, out_channels=1, kernel_size=5, stride=1, padding=2)
    
  
  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = self.dropout(x)
    x = F.relu(self.conv4(x))
    return x


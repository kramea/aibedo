from models.cnn import CNN
from models.ConvAutoencoder import ConvAutoencoder
import numpy as np
import torch
import torch.optim as optim
from torchsummary import summary
from data_loader import load_ncdf, normalize

#(1) Generate model
model = CNN()
#model = ConvAutoencoder()
#print model summary
input_shape = (1, 192, 288) # Input Size: (size of data, number of channels, lat, lon) 
summary(CNN(), input_shape)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0000005)

#(2) Load data
path = "/Users/sookim/Desktop/ALBEDO/aibedo/scripts/data/ours/rsut_Amon_CESM2_historical_r1i1p1f1_gn_185001-201412.nc"
dataset = load_ncdf(path)
dataset = normalize(dataset)

#(3) Train
model.train()
# number of epochs to train the model
n_epochs = 100
batch_size = 10
for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0
    for i in range(int(len(dataset)/batch_size)):
        # _ stands in for labels, here
        # no need to flatten images
        images = torch.tensor(dataset[i*batch_size: (i+1)*batch_size])

        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(images)
        # calculate the loss
        loss = criterion(outputs, images)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        #print("Minibatch step "+str(i)+" train loss "+str(loss.item())) 
        # update running training loss
        train_loss += loss.item()*images.size(0)
    # print avg training statistics 
    train_loss = train_loss/len(dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        train_loss
        ))

#Working okay
from models.cnn import CNN
import numpy as np
import os
import torch.optim as optim
from torchsummary import summary
from data_loader import load_ncdf, normalize, load_ncdf_to_SphereIcosahedral
from spherical_unet.models.spherical_unet.unet_model import SphericalUNet
from spherical_unet.utils.parser import create_parser, parse_config
from spherical_unet.utils.initialization import init_device
import torch
from spherical_unet.models.spherical_convlstm.convlstm import *
from spherical_unet.models.spherical_convlstm.convlstm_autoencoder import *
from spherical_unet.layers.samplings.icosahedron_pool_unpool import Icosahedron
from spherical_unet.utils.laplacian_funcs import get_equiangular_laplacians, get_healpix_laplacians, get_icosahedron_laplacians
from spherical_unet.layers.chebyshev import SphericalChebConv
import random


def temporal_conversion(data, time):
    """
       data: [T, N, C]
    """
    data = np.swapaxes(data, 1, 2)
    t,_,_ =np.shape(data)
    temporal_data = []
    stride = 1
    for i in range(0, int(t/stride)-time):
        d1,d2,d3 =np.shape(data[i*stride:i*stride+time])
        temporal_data.append( np.reshape(data[i*stride:i*stride+time], [1,d1,d2,d3]) )
    out = np.concatenate(temporal_data, axis=0)
    return out


def shuffle_data(d1, d2):
    n,_,_,_ = np.shape(d1)
    m,_,_,_ = np.shape(d2)
    assert(n == m)
    idx = [ i for i in range(m)]
    random.shuffle(idx)
    d1_out = []
    d2_out = []
    for i in range(n):
        d1_out.append(d1[i:i+1])
        d2_out.append(d2[i:i+1])
    d1_out2 = np.concatenate(d1_out, axis=0)
    d2_out2 = np.concatenate(d2_out, axis=0)
    return d1_out2, d2_out2



def main(parser_args):
    """Main function to create model and train, validation model.

    Args:
        parser_args (dict): parsed arguments
    """
    #(1) Generate model
    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    os.mkdir("./output_sunet/")
    model = SphericalConvLSTMAutoEncoder(parser_args.pooling_class, parser_args.n_pixels, parser_args.depth, parser_args.laplacian_type)
    print(model)
    model, device = init_device(parser_args.device, model)
    lr = parser_args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr)

    #print model summary and test
    n=parser_args.n_pixels
    # Run example
    out = model(torch.randn((3, 10, 5, n)))# Batch,Time,Channels,NumberofData
    
    print("output size", out.size())
    print(model)
    
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    #(2) Load data
    #un-comment below 3 lines if you want to resample spherical data from netcdf.
    #path = "/Users/sookim/Desktop/aibedo_sunet/aibedo/skeleton_framework/data/"
    #lon_list, lat_list, dataset = load_ncdf_to_SphereIcosahedral(path+"Processed_CESM2_r1i1p1f1_historical_Input.nc")
    #np.save("./data/input.npy",dataset)
    dataset = np.load("./data/Exp_All_CESM2_r1i1p1f1_historical_Input_5.npy")
    dataset = normalize(dataset, "in")
    #dataset = dataset[:,:,channel:channel+1]
    #lon_list, lat_list, dataset_out = load_ncdf_to_SphereIcosahedral(path+"Processed_CESM2_r1i1p1f1_historical_Output.nc")
    #np.save("./data/output.npy",dataset_out)
    dataset_out = np.load("./data/Exp_CESM2_r1i1p1f1_historical_Output_5.npy")
    dataset_out = normalize(dataset_out, "out")
    #channel = 2 #0,1,2
    timelength = 24
    n_epochs = 100
    batch_size = 2
    #dataset_out = dataset_out[:,:,channel:channel+1]
    print("input", np.shape(dataset),"output", np.shape(dataset_out))#(1980, 40962, 5) (1980, 40962, 3)

    #(2-2) Convert to temporal dataset
    dataset = temporal_conversion(dataset, timelength+1)
    dataset_out = temporal_conversion(dataset_out, timelength+1)
    # shuffle
    dataset, dataset_out = shuffle_data(dataset, dataset_out)
    print(np.shape(dataset), np.shape(dataset_out))
    #(3) Train test validation split: 80%/10%/10%
    n = len(dataset)
    #dataset_tr, dataset_te, dataset_va = dataset[0:int(0.90*n),0:-1,:,: ], dataset[int(0.90*n):int(0.95*n), 0:-1,:,:], dataset[int(0.95*n):, 0:-1, :,:]
    #dataset_out_tr, dataset_out_te, dataset_out_va = dataset_out[0:int(0.90*n),1:, :,:], dataset_out[int(0.90*n):int(0.95*n),1:, :,:], dataset_out[int(0.95*n):,1:, :,:]
    dataset_tr, dataset_te, dataset_va = dataset[0:int(0.90*n),0:-1,:,: ], dataset[int(0.90*n):int(0.95*n), 0:-1,:,:], dataset[int(0.95*n):, 0:-1, :,:]
    dataset_out_tr, dataset_out_te, dataset_out_va = dataset_out[0:int(0.90*n),:-1, :,:], dataset_out[int(0.90*n):int(0.95*n),:-1, :,:], dataset_out[int(0.95*n):,:-1, :,:]
    #(4) Train
    model.train()
    # number of epochs to train the model
    for epoch in range(0, n_epochs+1):
        # monitor training loss
        train_loss = 0.0
        for i in range(int(len(dataset_tr)/batch_size)):
            # _ stands in for labels, here
            # no need to flatten images
            images = torch.tensor(dataset_tr[i*batch_size: (i+1)*batch_size])
            gt_outputs = torch.tensor(dataset_out_tr[i*batch_size: (i+1)*batch_size])
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            #print(np.shape(images.float()))
            outputs = model(images.float())
            #print(outputs.float().size(),gt_outputs.float().size())
            # calculate the loss
            loss = criterion(outputs.float(), gt_outputs.float())
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            print("Minibatch step "+str(i)+" train loss "+str(loss.item()))
            # update running training loss
            train_loss += loss.item()*images.size(0)
        # print avg training statistics
        train_loss = train_loss/len(dataset)
        # validation
        with torch.no_grad():
            images = torch.tensor(dataset_va)
            gt_outputs = torch.tensor(dataset_out_va)
            outputs = model(images.float())
            validation_loss = criterion(outputs.float(), gt_outputs.float())
        print('Epoch: {} \tTraining Loss: {:.6f}\tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            validation_loss
            ))
        if epoch%1==0:
            if epoch == 0:
                os.mkdir("./saved_model_convlstm/")
            #save model
            torch.save(model.state_dict(), "./saved_model_convlstm/convlstm_state_"+str(epoch)+".pt")
            #test with testset
            with torch.no_grad():
                images = torch.tensor(dataset_te)
                gt_outputs = torch.tensor(dataset_out_te)
                outputs = model(images.float())
                test_loss = criterion(outputs.float(), gt_outputs.float())
                prediction = outputs.detach().numpy()
                groundtruth = gt_outputs.detach().numpy()
                print(np.shape(prediction), np.shape(groundtruth))
            np.save("./saved_model_convlstm/prediction_"+str(epoch)+"_"+str(test_loss)+".npy", prediction)
            np.save("./saved_model_convlstm/groundtruth_"+str(epoch)+"_"+str(test_loss)+".npy", groundtruth)

if __name__ == "__main__":
    PARSER_ARGS = parse_config(create_parser())
    main(PARSER_ARGS)

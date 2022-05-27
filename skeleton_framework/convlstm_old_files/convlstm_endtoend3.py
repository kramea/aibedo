from models.cnn import CNN
import numpy as np
import os, random, time, shutil
import torch
import torch.optim as optim
from torchsummary import summary
from data_loader import load_ncdf, normalize, load_ncdf_to_SphereIcosahedral, temporal_conversion, shuffle_data
from spherical_unet.utils.parser import create_parser, parse_config
from spherical_unet.utils.initialization import init_device
from spherical_unet.models.spherical_convlstm.convlstm import *
from spherical_unet.models.spherical_convlstm.convlstm_autoencoder import *


from spherical_unet.models.spherical_convlstm.convlstm import *
#from spherical_unet.models.spherical_convlstm.convlstm_unet import *
from spherical_unet.models.spherical_convlstm.convlstm_unet2 import *
from spherical_unet.layers.samplings.icosahedron_pool_unpool import Icosahedron
from spherical_unet.utils.laplacian_funcs import get_equiangular_laplacians, get_healpix_laplacians, get_icosahedron_laplacians
from spherical_unet.layers.chebyshev import SphericalChebConv
from spherical_unet.utils.samplings import icosahedron_nodes_calculator
from argparse import Namespace
from pathlib import Path


def main(parser_args):
    """Main function to create model and train, validation model.

    Args:
        parser_args (dict): parsed arguments
    """
    #(1) Generate model
    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
   
    print(parser_args)

    start = time.time()

    arg_input  = vars(parser_args)

    ## Module to check spherical gridded input file exists, if not generate for the grid level desired

    glevel = int(parser_args.depth)
    n_pixels = icosahedron_nodes_calculator(glevel)
    time_length = int(parser_args.time_length)

    print("Grid level:", glevel)
    print("N pixels:", n_pixels)
    print("time length:", time_length)

    temp_folder="./npy_files/" #Change this to where you want .npy files are saved
    # We don't want this as part of the github folder as the files can be large
    # Ideally we want to move this to S3 bucket

    infile = parser_args.input_file
    outfile = parser_args.output_file
    infname = Path(infile).stem
    outfname = Path(outfile).stem

    in_temp_npy_file = temp_folder + infname + "_" + str(glevel) + ".npy"
    out_temp_npy_file = temp_folder + outfname + "_" + str(glevel) + ".npy"
    print(in_temp_npy_file)
    print(out_temp_npy_file)
    

    in_channels = len(parser_args.input_vars)
    out_channels = len(parser_args.output_vars)

    if os.path.exists(in_temp_npy_file):
        print("Gridded input .npy file exists")
    else:
        print("Generating input .npy file ", in_temp_npy_file, "at grid level", glevel, "...")
        lon_list, lat_list, dset = load_ncdf_to_SphereIcosahedral(infile, glevel, parser_args.input_vars)
        np.save(in_temp_npy_file, dset)
    
    if os.path.exists(out_temp_npy_file):
        print("Gridded output .npy file exists")
    else:
        print("Generating output .npy file ", out_temp_npy_file, "at grid level", glevel, "...")
        lon_list, lat_list, dset = load_ncdf_to_SphereIcosahedral(outfile, glevel, parser_args.output_vars)
        np.save(out_temp_npy_file, dset)

    if not parser_args.generation_only:    
    
        output_path = parser_args.output_path
    
        if os.path.isdir(output_path):
            shutil.rmtree(output_path)
    
        os.mkdir(output_path)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = SphericalConvLSTMAutoEncoder(parser_args.pooling_class, n_pixels, 6, parser_args.laplacian_type,8,3)
        #SphericalConvLSTMUnet(parser_args.pooling_class, n_pixels, 6 , parser_args.laplacian_type, in_channels, out_channels)
        model = model.to(device)
        model, device = init_device(parser_args.device, model)
        lr = parser_args.learning_rate
        optimizer = optim.Adam(model.parameters(), lr=lr)
        #print(model)

        criterion = torch.nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        #(2) Load data
        dataset = np.load(in_temp_npy_file)
        print(np.shape(dataset))
        dataset = normalize(dataset, "in")
        dataset_out = np.load(out_temp_npy_file)
        dataset_out = normalize(dataset_out, "out")
        

        dataset = temporal_conversion(dataset, time_length)
        dataset_out = temporal_conversion(dataset_out, time_length)
        # shuffle
        dataset, dataset_out = shuffle_data(dataset, dataset_out)
        # collect only last timestep from output
        dataset = dataset[:,-1:,:,:]
        dataset_out = dataset_out[:,-1:,:,:]

        print("Timelength of input: "+str(time_length))
        print("Shape: (1) Input ",np.shape(dataset),"(2) Output ", np.shape(dataset_out))
        #(3) Train test validation split: 80%/10%/10%
        n = len(dataset)
        dataset_tr, dataset_te, dataset_va = dataset[0:int(0.8*n)], dataset[int(0.8*n):int(0.9*n)], dataset[int(0.9*n):]
        dataset_out_tr, dataset_out_te, dataset_out_va = dataset_out[0:int(0.8*n)], dataset_out[int(0.8*n):int(0.9*n)], dataset_out[int(0.9*n):]
        #(4) Train 
        model.train()
        # number of epochs to train the model
        n_epochs = parser_args.n_epochs
        batch_size = parser_args.batch_size
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
                outputs = model(images.float())
                
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
            if epoch%2==0:
                if epoch == 0:
                    os.mkdir("./model_convlstmautoencoder_exp7_"+str(time_length)+"/")
                #save model
                torch.save(model.state_dict(), "./model_convlstmautoencoder_exp7_"+str(time_length)+"/convlstm_state_"+str(epoch)+".pt")
                
                #test with testset
                with torch.no_grad():
                    images = torch.tensor(dataset_te)
                    gt_outputs = torch.tensor(dataset_out_te)
                    outputs = model(images.float())
                    test_loss = criterion(outputs.float(), gt_outputs.float())
                    prediction = outputs.detach().numpy()
                    groundtruth = gt_outputs.detach().numpy()
                np.save("./model_convlstmautoencoder_exp7_"+str(time_length)+"/prediction_"+str(epoch)+"_"+str(test_loss)+".npy", prediction)
                np.save("./model_convlstmautoencoder_exp7_"+str(time_length)+"/groundtruth_"+str(epoch)+"_"+str(test_loss)+".npy", groundtruth)

        end = time.time()
        print(f"Runtime of the program is {end - start}")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    PARSER_ARGS = parse_config(create_parser())
    main(PARSER_ARGS)


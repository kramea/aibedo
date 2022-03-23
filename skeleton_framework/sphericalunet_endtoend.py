from models.cnn import CNN
import numpy as np
import os, shutil
import torch
import torch.optim as optim
from torchsummary import summary
from data_loader import load_ncdf, normalize, load_ncdf_to_SphereIcosahedral
from spherical_unet.models.spherical_unet.unet_model import SphericalUNet
from spherical_unet.utils.parser import create_parser, parse_config
from spherical_unet.utils.initialization import init_device
from spherical_unet.utils.samplings import icosahedron_nodes_calculator
from argparse import Namespace
from pathlib import Path
import time


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
    lag = int(parser_args.time_lag)


    print("Grid level:", glevel)
    print("N pixels:", n_pixels)
    print("time lag:", lag)

    temp_folder="./npy_files/" #Change this to where you want .npy files are saved
    # We don't want this as part of the github folder as the files can be large
    # Ideally we want to move this to S3 bucket

    #infile = "/home/ubuntu/Exp2_CESM2_r1i1p1f1_historical_Input.nc"
    infile = parser_args.input_file
    #outfile = "/home/ubuntu/Exp1_CESM2_r1i1p1f1_historical_Output.nc"
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
        unet = SphericalUNet(parser_args.pooling_class, n_pixels, 6, parser_args.laplacian_type, parser_args.kernel_size,in_channels, out_channels)
        unet = unet.to(device)
        # It only works for level 6--why? (the above line)<--Kalai come back to this
        unet, device = init_device(parser_args.device, unet)
        lr = parser_args.learning_rate
        optimizer = optim.Adam(unet.parameters(), lr=lr)

        print(unet)

        #print model summary and test
        #out = unet(torch.randn(10, 10242 ,5))
        #print(unet, out.shape)
        criterion = torch.nn.MSELoss()
        optimizer = optim.Adam(unet.parameters(), lr=lr)

        #(2) Load data
        #un-comment below 3 lines if you want to resample spherical data from netcdf.
        #lon_list, lat_list, dataset = load_ncdf_to_SphereIcosahedral(path+"MPI_ESM1_2_LR_r1i1p1f1_historical_Input.nc")
        #np.save("./data/input_new_5.npy",dataset)
        #dataset = np.load(parser_args.input_file)
        dataset = np.load(in_temp_npy_file)
        #dataset = normalize(dataset, "in")--Question to Soo from Kalai: this was uncommented. Are we normalizing again?
        
        #lon_list, lat_list, dataset_out = load_ncdf_to_SphereIcosahedral(path+"MPI_ESM1_2_LR_r1i1p1f1_historical_Output.nc")
        #np.save("./data/output_new_5.npy",dataset_out)
        #dataset_out = np.load(parser_args.output_file)
    
        dataset_out = np.load(out_temp_npy_file)
        #dataset_out = normalize(dataset_out, "out")--Same here?
        
        print(np.shape(dataset), np.shape(dataset_out))
        if lag > 0:
            dataset = dataset[:-lag]
            dataset_out = dataset_out[lag:]
        print(np.shape(dataset), np.shape(dataset_out))
        #(3) Train test validation split: 80%/10%/10%
        n = len(dataset)
        dataset_tr, dataset_te, dataset_va = dataset[0:int(0.8*n)], dataset[int(0.8*n):int(0.9*n)], dataset[int(0.9*n):]
        dataset_out_tr, dataset_out_te, dataset_out_va = dataset_out[0:int(0.8*n)], dataset_out[int(0.8*n):int(0.9*n)], dataset_out[int(0.9*n):]
        #(4) Train
        
        unet.train()
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
                outputs = unet(images.float())
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
                outputs = unet(images.float())
                validation_loss = criterion(outputs.float(), gt_outputs.float())
            print('Epoch: {} \tTraining Loss: {:.6f}\tValidation Loss: {:.6f}'.format(
                epoch,
                train_loss,
                validation_loss
                ))
            if epoch%5==0:
                if epoch == 0:
                    os.mkdir("./saved_model/")
                #save model
                torch.save(unet.state_dict(), "./saved_model/unet_state_"+str(epoch)+".pt")
                #test with testset
                with torch.no_grad():
                    images = torch.tensor(dataset_te)
                    gt_outputs = torch.tensor(dataset_out_te)
                    outputs = unet(images.float())
                    test_loss = criterion(outputs.float(), gt_outputs.float())
                    prediction = outputs.detach().numpy()
                    groundtruth = gt_outputs.detach().numpy()
                np.save("./saved_model/prediction_"+str(epoch)+"_"+str(test_loss)+".npy", prediction)
                np.save("./saved_model/groundtruth_"+str(epoch)+"_"+str(test_loss)+".npy", groundtruth)

        end = time.time()
        print(f"Runtime of the program is {end - start}")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    PARSER_ARGS = parse_config(create_parser())
    main(PARSER_ARGS)


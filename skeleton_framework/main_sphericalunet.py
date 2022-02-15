from models.cnn import CNN
import numpy as np
import os
import torch
import torch.optim as optim
from torchsummary import summary
from data_loader import load_ncdf, normalize, load_ncdf_to_SphereIcosahedral
from spherical_unet.models.spherical_unet.unet_model import SphericalUNet
from spherical_unet.utils.parser import create_parser, parse_config
from spherical_unet.utils.initialization import init_device

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
    unet = SphericalUNet(parser_args.pooling_class, parser_args.n_pixels, parser_args.depth, parser_args.laplacian_type, parser_args.kernel_size)
    unet, device = init_device(parser_args.device, unet)
    lr = parser_args.learning_rate
    optimizer = optim.Adam(unet.parameters(), lr=lr)

    #print model summary and test
    out = unet(torch.randn(10, 10242,5))
    print(unet, out.shape)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(unet.parameters(), lr=lr)

    #(2) Load data
    #un-comment below 3 lines if you want to resample spherical data from netcdf.
    path = "/Users/sookim/aibedo/skeleton_framework/data/"
    lon_list, lat_list, dataset = load_ncdf_to_SphereIcosahedral(path+"Processed_CESM2_r1i1p1f1_historical_Input.nc")
    np.save("./data/input.npy",dataset)
   # dataset = np.load("./data/SphereIcosahedral_rsut_Amon_CESM2_historical_r1i1p1f1_gn_185001-201412_1.npy")
    dataset = normalize(dataset, "in")
    lon_list, lat_list, dataset_out = load_ncdf_to_SphereIcosahedral(path+"Processed_CESM2_r1i1p1f1_historical_Output.nc")
    np.save("./data/output.npy",dataset_out)
    dataset_out = normalize(dataset_out, "out")
    print(np.shape(dataset_out))
    #(3) Train
    unet.train()
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
            gt_outputs = torch.tensor(dataset_out[i*batch_size: (i+1)*batch_size])
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
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch,
            train_loss
            ))




if __name__ == "__main__":
    PARSER_ARGS = parse_config(create_parser())
    main(PARSER_ARGS)

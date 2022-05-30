from models.cnn import CNN
import numpy as np
import xarray as xr
import os, shutil
import torch
import torch.optim as optim
from torchsummary import summary
from data_loader import load_ncdf, normalize, load_ncdf_to_SphereIcosahedral, shuffle_data
# from spherical_unet.models.spherical_unet.unet_model import SphericalUNet
from spherical_unet.utils.parser import create_parser, parse_config
from spherical_unet.utils.initialization import init_device
from spherical_unet.utils.samplings import icosahedron_nodes_calculator
from argparse import Namespace
from pathlib import Path
import time

from ignite.contrib.handlers.param_scheduler import create_lr_scheduler_with_warmup
from ignite.contrib.handlers.tensorboard_logger import GradsHistHandler, OptimizerParamsHandler, OutputHandler, \
    TensorboardLogger, WeightsHistHandler
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.handlers import EarlyStopping, TerminateOnNan
from ignite.metrics import EpochMetric, Accuracy, Loss
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

def sunet_collate(batch):

    batchShape = batch[0].shape
    varlimit = batchShape[1] - 3  # 3 output variables: tas, psl, pr
    
    data_in_array = np.array([item[:, 0:varlimit] for item in batch])
    data_out_array = np.array([item[:, varlimit:] for item in batch])
    
    data_in = torch.Tensor(data_in_array)
    data_out = torch.Tensor(data_out_array)
    return [data_in, data_out]


def get_dataloader(parser_args):

    glevel = int(parser_args.depth)
    n_pixels = icosahedron_nodes_calculator(glevel)
    time_length = int(parser_args.time_length)

    print("Grid level:", glevel)
    print("N pixels:", n_pixels)
    print("time length:", time_length)

    inDS = xr.open_dataset(parser_args.input_file)
    outDS = xr.open_dataset(parser_args.output_file)

    lon_list = inDS.lon.data
    lat_list = inDS.lat.data

    in_channels = len(parser_args.input_vars)
    out_channels = len(parser_args.output_vars)


    #Input data
    data_all = []
    for var in parser_args.input_vars:
        temp_data = np.reshape(np.concatenate(inDS[var].data, axis = 0), [-1,n_pixels,1])
        data_all.append(temp_data)
    dataset_in = np.concatenate(data_all, axis=2)

    months = np.resize(np.arange(12), 12 * 165)  # 1980/12 = 165
    #Output data
    data_all = []
    for var in parser_args.output_vars:
        temp_data = np.reshape(np.concatenate(outDS[var].data, axis = 0), [-1, n_pixels,1])
        data_all.append(temp_data)
    dataset_out = np.concatenate(data_all, axis=2)

    h = np.concatenate((dataset_out, months), axis=0)
    print(h.shape)





    #print(dataset_out.shape)

    #months = list(np.resize(np.arange(12), 12*165)) #1980/12 = 165
    #dataset_out_mo = np.expand_dims(dataset_out, axis = 0)
    #print(dataset_out_mo.shape)
    #dataset_out_mo = np.concatenate((dataset_out_mo, months), axis = 3)

    #print(dataset_out_mo.shape)


    #dataset_out = np.concatenate((dataset_out, np.zeros(dataset_out.shape[0]).reshape(-1,1)), axis = 1)
    #dataset_out = oneHotEncode3D(dataset_out)


    dataset_in, dataset_out = shuffle_data(dataset_in, dataset_out)


    #print(dataset_out[:,:,0])



    if parser_args.time_lag > 0:
        dataset_in = dataset_in[:-parser_args.time_lag]
        dataset_out = dataset_out[parser_args.time_lag:]

    combined_data = np.concatenate((dataset_in, dataset_out), axis=2)

    train_data, temp = train_test_split(combined_data, train_size=parser_args.partition[0], random_state=43)
    val_data, test_data = train_test_split(temp, test_size=parser_args.partition[2] / (
                parser_args.partition[1] + parser_args.partition[2]), random_state=43)

    dataloader_train = DataLoader(train_data, batch_size=parser_args.batch_size, shuffle=True, num_workers=12, collate_fn=sunet_collate)
    dataloader_validation = DataLoader(val_data, batch_size=parser_args.batch_size, shuffle=False, num_workers=12, collate_fn=sunet_collate)
    dataloader_test = DataLoader(test_data, batch_size=parser_args.batch_size, shuffle=False, num_workers=12, collate_fn=sunet_collate)
    return dataloader_train, dataloader_validation, dataloader_test

def main(parser_args):
    """Main function to create model and train, validation model.
    Args:
        parser_args (dict): parsed arguments
    """
    # (1) Generate model
    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel

    dataloader_train, dataloader_validation, dataloader_test = get_dataloader(parser_args)

    criterion = torch.nn.MSELoss()

    print("Dataloader train size", len(dataloader_train.dataset[0].shape))

    output_path = parser_args.output_path

    if os.path.isdir(output_path):
        shutil.rmtree(output_path)

    os.mkdir(output_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_pixels = icosahedron_nodes_calculator(parser_args.depth)
    if parser_args.depth > 4:
        print("Generating 6 layered unet")
        # for glevel = 5,6 --> use 6-layered unet
        from spherical_unet.models.spherical_unet.unet_model import SphericalUNet
        unet = SphericalUNet(parser_args.pooling_class, n_pixels, 6, parser_args.laplacian_type,
                             parser_args.kernel_size, len(parser_args.input_vars), len(parser_args.output_vars))
    else:
        print("Generating 3 layered unet")
        # for glevel = 1,2,3,4 --> use 3-layered unet (shllow)
        from spherical_unet.models.spherical_unet_shallow.unet_model import SphericalUNet
        unet = SphericalUNet(parser_args.pooling_class, n_pixels, 3, parser_args.laplacian_type,
                             parser_args.kernel_size, len(parser_args.input_vars), len(parser_args.output_vars))

    print(unet)
    # unet = unet.to(device)
    unet, device = init_device(parser_args.device, unet)
    lr = parser_args.learning_rate

    optimizer = optim.Adam(unet.parameters(), lr=lr)

    def trainer(engine, batch):
        unet.train()
        data_in, data_out = batch
        data_in = data_in.to(device)
        data_out = data_out.to(device)
        optimizer.zero_grad()
        outputs = unet(data_in)
        # function called next to function definition 
        normVal = normVal.to(device)
        data_out_unscaled = torch.zeros_like(data_out)
        outputs_unscaled = torch.zeros_like(outputs)

        '''
        Note: Currently order of output is assumed to tas, psl, pr
              If that is incorrect, please make edits in the section below or 
              refer to the function at the end.
        '''
        for i in range(data_out.shape[0]):
            # store month encoding in variable idx. 
            # use it to access value from mean/std file 
            # all fields will have same encoding
            idx = data_out[i,-1,0]
            
            # for ground truth data

            data_out_unscaled[i,0:int(data_out.shape[1])-1, 0] = data_out[i,0:int(data_out.shape[1])-1, 0] * normVal[int(idx),:, 1] + normVal[int(idx),:, 0]
            data_out_unscaled[i,0:int(data_out.shape[1])-1, 1] = data_out[i,0:int(data_out.shape[1])-1, 1] * normVal[int(idx),:, 3] + normVal[int(idx),:, 2]
            data_out_unscaled[i,0:int(data_out.shape[1])-1, 2] = data_out[i,0:int(data_out.shape[1])-1, 2] * normVal[int(idx),:, 5] + normVal[int(idx),:, 4]
    
            # for outputs

            outputs_unscaled[i, :, 0] = outputs[i, :, 0] * normVal[int(idx),:, 1] + normVal[int(idx),:, 0]
            outputs_unscaled[i, :, 1] = outputs[i, :, 1] * normVal[int(idx),:, 3] + normVal[int(idx),:, 2]
            outputs_unscaled[i, :, 2] = outputs[i, :, 2] * normVal[int(idx),:, 5] + normVal[int(idx),:, 4]
        
        outputs = precip_pos(outputs_unscaled)
        data_out = data_out[:,0:int(data_out.shape[1])-1, :] # removing the extra dimension of one_hot encoding
        loss = criterion(outputs.float(), data_out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    engine_train = Engine(trainer)

    val_metrics = {
        "mse": Loss(criterion)
    }

    evaluator = create_supervised_evaluator(unet, metrics=val_metrics, device=device)

    engine_train.add_event_handler(Events.EPOCH_STARTED, lambda x: print("Starting Epoch: {}".format(x.state.epoch)))

    @engine_train.on(Events.ITERATION_COMPLETED(every=10))
    def log_training_results_iteration(engine):
        evaluator.run(dataloader_train)
        metrics = evaluator.state.metrics
        print(
            f"Training Results - Iteration: {engine_train.state.iteration}  Avg loss: {metrics['mse']:.4f}")

    @engine_train.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(dataloader_train)
        metrics = evaluator.state.metrics
        print(
            f"Training Results - Epoch: {engine_train.state.epoch}  Avg loss: {metrics['mse']:.4f}")


    @engine_train.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(dataloader_validation)
        metrics = evaluator.state.metrics
        print(
            f"Validation Results - Epoch: {engine_train.state.epoch} Avg loss: {metrics['mse']:.4f}")

    engine_train.run(dataloader_train, max_epochs=parser_args.n_epochs)

    saved_model_path = "./saved_model_lag_" + str(parser_args.time_lag)
    if os.path.isdir(saved_model_path):
        shutil.rmtree(saved_model_path)
        os.mkdir(saved_model_path)
    else:
        os.mkdir(saved_model_path)

    torch.save(unet.state_dict(), "./saved_model_lag_" + str(parser_args.time_lag) + "/unet_state_" + str(parser_args.n_epochs) + ".pt")

    # Prediction code

    #model.load_state_dict(model.state_dict())
    unet.eval()

    predictions = np.empty((parser_args.batch_size, n_pixels, len(parser_args.output_vars)))
    groundtruth = np.empty((parser_args.batch_size, n_pixels, len(parser_args.output_vars)))
    for batch in dataloader_test:
        data_in, data_out = batch
        preds = unet(data_in)
        pred_numpy = preds.detach().cpu().numpy()
        predictions = np.concatenate((predictions, pred_numpy), axis=0)
        groundtruth = np.concatenate((groundtruth, data_out.detach().cpu().numpy()), axis=0)

    np.save("./saved_model_lag_" + str(parser_args.time_lag) + "/prediction_"+str(parser_args.n_epochs)+".npy", predictions)
    np.save("./saved_model_lag_" + str(parser_args.time_lag) + "/groundtruth_"+str(parser_args.n_epochs)+".npy", groundtruth)

'''
def oneHotEncode3D(field):
    
    q, j = 0, 0
    
    for i in range(field.shape[0]):
        
        if i == 0:
            
            field[j, -1, :] = i
        
        else:
            q = i // 12
            
            i = i - q * 12
            
            j = j + 1
            
            field[j, -1, :] = i

    return field '''

def precip_pos(output):

    output = output.detach().numpy()

    precip = np.array(output[:, :, -1])

    precip[precip<0] = 0 # set any negative values to zero
    
    output[:, :, -1] = precip # update original precip value to reflect updated array
    
    '''
    
    Note: rescaling it back to the data space : (unscaled - mean) / std 

    '''

    for i in range(output.shape[0]):
        
        idx = output[i,-1,0]

        output[i, :, 2] = (output[i, :, 2] - normVal[int(idx),:, 4]) / normVal[int(idx),:, 5]
    
    output = output.to(device)

    return output

'''
Need to update path!
'''

foutput_mean = "/data-ssd/kramea/data_aibedo/ymonmean.1980_2010.isosph.CMIP6.historical.ensmean.Output.nc"

foutput_std = "/data-ssd/kramea/data_aibedo/ymonstd.1980_2010.isosph.CMIP6.historical.ensmean.Output.nc"

def loadVal(foutput_mean, foutput_std):

    dsoutput_mean = xr.open_dataset(foutput_mean)
    
    dsoutput_std = xr.open_dataset(foutput_std)
    
    tas_mean = np.array(dsoutput_mean.tas_pre)
    
    pr_mean = np.array(dsoutput_mean.pr_pre)
    
    psl_mean = np.array(dsoutput_mean.psl_pre)
    
    tas_std = np.array(dsoutput_std.tas_pre)
    
    pr_std = np.array(dsoutput_std.pr_pre)
    
    psl_std = np.array(dsoutput_std.psl_pre)
    
    normVal = np.dstack((tas_mean, tas_std, psl_mean, psl_std, pr_mean, pr_std))
                        
    return normVal

# Read into memory just once as this value wont change during one single experimentation

normVal = loadVal(foutput_mean, foutput_std)

if __name__ == "__main__":
    PARSER_ARGS = parse_config(create_parser())
    main(PARSER_ARGS)


from models.cnn import CNN
import numpy as np
import xarray as xr
import os, shutil
import torch
import torch.optim as optim
from torchsummary import summary
from data_loader import load_ncdf, normalize, load_ncdf_to_SphereIcosahedral, shuffle_data, shuffle_data_meanstd
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
from ignite.engine import Engine, Events, create_supervised_evaluator, create_supervised_trainer
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import EarlyStopping, TerminateOnNan
from ignite.metrics import EpochMetric, Accuracy, Loss
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from ignite.utils import convert_tensor
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


def sunet_collate(batch):
    batchShape = batch[0].shape
    varlimit = batchShape[1] - 3  # 3 output variables: tas, psl, pr, 3 mean, 3 std

    data_in_array = np.array([item[:, 0:varlimit] for item in batch])  # includes mean and std
    # data_out_array = np.array([item[:, varlimit:] for item in batch])
    data_out_array = np.array([item[:, varlimit:] for item in batch])
    #data_mean_array = np.array([item[:, varlimit-6:varlimit-3] for item in batch])
    # data_std_array = np.array([item[:, varlimit + 6:] for item in batch])

    data_in = torch.Tensor(data_in_array)
    data_out = torch.Tensor(data_out_array)
    #data_mean = torch.Tensor(data_mean_array)
    # data_std = torch.Tensor(data_std_array)
    return [data_in, data_out]
    #return [data_in_array, data_out_array]



def get_dataloader(parser_args):
    glevel = int(parser_args.depth)
    n_pixels = icosahedron_nodes_calculator(glevel)
    time_length = int(parser_args.time_length)

    print("Grid level:", glevel)
    print("N pixels:", n_pixels)
    print("time length:", time_length)

    inDS = xr.open_dataset(parser_args.input_file)
    outDS = xr.open_dataset(parser_args.output_file)
    meanDS = xr.open_dataset(parser_args.mean_file)
    stdDS = xr.open_dataset(parser_args.std_file)
    peErr = xr.open_dataset(parser_args.pe_err)
    psErr = xr.open_dataset(parser_args.ps_err)
    peStd = xr.open_dataset(parser_args.pe_std)
    psStd = xr.open_dataset(parser_args.ps_std)


    lon_list = inDS.lon.data
    lat_list = inDS.lat.data

    in_channels = len(parser_args.input_vars)
    out_channels = len(parser_args.output_vars)


    # Dictionary of Monthly Precipitation

    month = np.arange(12)

    #Pixel-wise month (assigned to each grid)

    month_pixel_data = np.reshape(np.repeat(month, n_pixels), [-1, n_pixels, 1])

    # Input data
    data_all = []
    for var in parser_args.input_vars:
        temp_data = np.reshape(np.concatenate(inDS[var].data, axis=0), [-1, n_pixels, 1])
        data_all.append(temp_data)


    data_month = []
    for c in range(165):
        data_month.append(month_pixel_data)

    dataset_month = np.concatenate(data_month, axis=0)

    data_all.append(dataset_month)
    dataset_in = np.concatenate(data_all, axis=2)

    print("dataset with month", dataset_in.shape)

    # Output data
    data_all = []
    for var in parser_args.output_vars:
        temp_data = np.reshape(np.concatenate(outDS[var].data, axis=0), [-1, n_pixels, 1])
        data_all.append(temp_data)
    dataset_out = np.concatenate(data_all, axis=2)

    meanPr = meanDS.pr.data
    pr_mean_dict = {}
    for m, p in zip(month, meanPr):
        pr_mean_dict[m] = p

    stdPr = stdDS.pr.data
    pr_std_dict = {}
    for m, p in zip(month, stdPr):
        pr_std_dict[m] = p

    meanPs = meanDS.ps.data
    ps_mean_dict = {}
    for m, p in zip(month, meanPs):
        ps_mean_dict[m] = p

    stdPs = stdDS.ps.data
    ps_std_dict = {}
    for m, p in zip(month, stdPs):
        ps_std_dict[m] = p

    # Pe_err/std, Ps_err/std terms
    # peErr = stdDS.ps.data

    pe_err_dict = {}
    for m, p in zip(month, peErr:
        pe_err_dict[m] = p
    '''
    pe_errstd_dict = {}
    for m, p in zip(month, peStd:
        pe_errstd_dict[m] = p
    '''
    ps_err_dict = {}
    for m, p in zip(month, psErr:
        ps_err_dict[m] = p
    '''
    ps_errstd_dict = {}
    for m, p in zip(month, psStd:
        ps_errstd_dict[m] = p
    '''

    combined_data = np.concatenate((dataset_in, dataset_out), axis=2)

    train_data, temp = train_test_split(combined_data, train_size=parser_args.partition[0], random_state=43)
    val_data, test_data = train_test_split(temp, test_size=parser_args.partition[2] / (
            parser_args.partition[1] + parser_args.partition[2]), random_state=43)

    dataloader_train = DataLoader(train_data, batch_size=parser_args.batch_size, shuffle=True, num_workers=12,
                                  collate_fn=sunet_collate)
    dataloader_validation = DataLoader(val_data, batch_size=parser_args.batch_size, shuffle=False, num_workers=12,
                                       collate_fn=sunet_collate)
    dataloader_test = DataLoader(test_data, batch_size=parser_args.batch_size, shuffle=False, num_workers=12,
                                 collate_fn=sunet_collate)
    return dataloader_train, dataloader_validation, dataloader_test, pr_mean_dict, pr_std_dict, ps_mean_dict, ps_std_dict, pe_err_dict, ps_err_dict


def main(parser_args):
    """Main function to create model and train, validation model.
    Args:
        parser_args (dict): parsed arguments
    """
    # (1) Generate model
    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel

    dataloader_train, dataloader_validation, dataloader_test, pr_mean_dict, pr_std_dict, ps_mean_dict, ps_std_dict, \
    pe_err_dict, ps_err_dict = get_dataloader(parser_args)

    writer = SummaryWriter("./")

    criterion = torch.nn.MSELoss() # why two losses?

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
                             parser_args.kernel_size, len(parser_args.input_vars)+1, len(parser_args.output_vars))
    else:
        print("Generating 3 layered unet")
        # for glevel = 1,2,3,4 --> use 3-layered unet (shllow)
        from spherical_unet.models.spherical_unet_shallow.unet_model import SphericalUNet
        unet = SphericalUNet(parser_args.pooling_class, n_pixels, 3, parser_args.laplacian_type,
                             parser_args.kernel_size, len(parser_args.input_vars)+1, len(parser_args.output_vars))

    #print(unet)
    # unet = unet.to(device)
    unet, device = init_device(parser_args.device, unet)
    lr = parser_args.learning_rate
    loss = torch.nn.MSELoss()
    optimizer = optim.Adam(unet.parameters(), lr=lr)

    '''trainer = create_supervised_trainer(unet, optimizer=optimizer, loss_fn=criterion, prepare_batch=custom_prepare_batch)'''

    def trainer(engine, batch):

        data_in, data_out = batch
        batch_month = [[m[0] for m in np.array(data_in[:,:,7])]]
        # for precipitation
        pr_data_mean = [pr_mean_dict[k] for k in batch_month[0]]
        pr_data_std = [pr_std_dict[k] for k in batch_month[0]]
        # for surface pressure
        ps_data_mean = [ps_mean_dict[k] for k in batch_month[0]]
        ps_data_std = [ps_std_dict[k] for k in batch_month[0]]
        # for PsErr
        PE_Err = [pe_err_dict[k] for k in batch_month[0]]
        #for PeErr
        PS_Err = [ps_err_dict[k] for k in batch_month[0]]

        # unscaling fields
        unscaled_data_out_pr = (np.array(data_out[:,:,2]) * np.array(pr_data_std)) + pr_data_mean
        unscaled_data_out_ps = (np.array(data_out[:, :, 1]) * np.array(ps_data_std)) + ps_data_mean
        # where is the evpsbl (evaporation) -- Ask Kalai

        data_in = data_in.to(device)
        data_out = data_out.to(device)

        optimizer.zero_grad()
        unet.train()
        outputs = unet(data_in)

        '''
        Added section begins
        '''

        outputs_detach = outputs.detach().cpu().numpy()
        
        ## constraint 4 - precipitation constraint
        
        outputs_unscaled_pr = (np.array(outputs_detach[:,:,2]) * data_std) + data_mean

        outputs_unscaled_pr[outputs_unscaled_pr < 0] = 0

        ## constraint 3 - global moisture constraint
        # average monthly and then subtract from PE_Err
        ## constraint 5 - mass conservation constraint
        # average monthly and then subtract from PS_Err
        # assumption: the order in outputs are aligned to the order in the batch/mean
        for i in range(batch_month[0].shape):
            loss_pr[i,:] = np.mean(outputs_unscaled_pr[i,:]) - PE_Err[i] # need to subtract evaporation
            loss_ps[i,:] = np.mean(outputs_unscaled_ps[i,:] - PS_Err[i]
        
        # rescaling needed for other terms as well.
        outputs_rescaled_pr =  ((outputs_unscaled_pr - pr_data_mean) / pr_data_std) # some regularizer
        
        # normalize
        outputs[:, :, 2] = torch.from_numpy(outputs_rescaled_pr).to(outputs)
        
        # update a new loss function with adding constraints
        
        loss_constraints = np.mean(loss_pr) + np.mean(loss_ps) # check in with Kalai what to do about this
        
        '''
        Added section ends
        '''

        loss_mse = criterion(outputs.float(), data_out)

        # updated loss value

        loss = loss_mse + loss_constraints

        print("loss", loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    engine_train = Engine(trainer)



    val_metrics = {
        "mse": Loss(criterion)
    }

    evaluator = create_supervised_evaluator(unet, metrics=val_metrics, device=device)

    #engine_train.add_event_handler(Events.EPOCH_STARTED, lambda x: print("Starting Epoch: {}".format(x.state.epoch)))

    '''@engine_train.on(Events.ITERATION_COMPLETED(every=10))
    def log_training_results_iteration(engine):
        evaluator.run(dataloader_train)
        metrics = evaluator.state.metrics
        print(
            f"Training Results - Iteration: {engine_train.state.iteration}  Avg loss: {metrics['mse']:.4f}")'''

    @engine_train.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(dataloader_train)
        metrics = evaluator.state.metrics
        print(
            f"Training Results - Epoch: {engine_train.state.epoch}  Avg loss: {metrics['mse']:.4f}")
        writer.add_scalars("Loss/train", metrics, engine_train.state.epoch)
        #writer.close()

    @engine_train.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(dataloader_validation)
        metrics = evaluator.state.metrics
        print(
            f"Validation Results - Epoch: {engine_train.state.epoch} Avg loss: {metrics['mse']:.4f}")
        writer.add_scalars("Loss/validation", metrics, engine_train.state.epoch)
        writer.close()

    pbar = ProgressBar()
    pbar.attach(engine_train, output_transform=lambda x: {"loss": x})
    engine_train.run(dataloader_train, max_epochs=parser_args.n_epochs)


    saved_model_path = "./saved_model_lag_" + str(parser_args.time_lag)
    if os.path.isdir(saved_model_path):
        shutil.rmtree(saved_model_path)
        os.mkdir(saved_model_path)
    else:
        os.mkdir(saved_model_path)

    torch.save(unet.state_dict(),
               "./saved_model_lag_" + str(parser_args.time_lag) + "/unet_state_" + str(parser_args.n_epochs) + ".pt")

    # Prediction code

    # model.load_state_dict(model.state_dict())
    unet.eval()

    predictions = np.empty((parser_args.batch_size, n_pixels, len(parser_args.output_vars)))
    groundtruth = np.empty((parser_args.batch_size, n_pixels, len(parser_args.output_vars)))
    for batch in dataloader_test:
        data_in, data_out = batch
        preds = unet(data_in)
        pred_numpy = preds.detach().cpu().numpy()
        predictions = np.concatenate((predictions, pred_numpy), axis=0)
        groundtruth = np.concatenate((groundtruth, data_out.detach().cpu().numpy()), axis=0)

    np.save("./saved_model_lag_" + str(parser_args.time_lag) + "/prediction_" + str(parser_args.n_epochs) + ".npy",
            predictions)
    np.save("./saved_model_lag_" + str(parser_args.time_lag) + "/groundtruth_" + str(parser_args.n_epochs) + ".npy",
            groundtruth)

if __name__ == "__main__":
    PARSER_ARGS = parse_config(create_parser())
    main(PARSER_ARGS)


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

    data_in_array = np.array([item[:, 0:varlimit-6] for item in batch])  # includes mean and std
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

    lon_list = inDS.lon.data
    lat_list = inDS.lat.data

    in_channels = len(parser_args.input_vars)
    out_channels = len(parser_args.output_vars)

    # Input data
    data_all = []
    for var in parser_args.input_vars:
        temp_data = np.reshape(np.concatenate(inDS[var].data, axis=0), [-1, n_pixels, 1])
        data_all.append(temp_data)
    dataset_in = np.concatenate(data_all, axis=2)

    # Output data
    data_all = []
    for var in parser_args.output_vars:
        temp_data = np.reshape(np.concatenate(outDS[var].data, axis=0), [-1, n_pixels, 1])
        data_all.append(temp_data)
    dataset_out = np.concatenate(data_all, axis=2)

    # Mean data

    data_all = []
    for var in parser_args.meanstd_vars:
        temp_data = np.reshape(np.concatenate(meanDS[var].data, axis=0), [-1, n_pixels, 1])
        data_all.append(temp_data)
    dataset_mean_v1 = np.concatenate(data_all, axis=2)

    data_all = []
    for c in range(int(len(dataset_out) / len(dataset_mean_v1))):
        data_all.append(dataset_mean_v1)

    dataset_mean = np.concatenate(data_all, axis=0)

    # Std data

    data_all = []
    for var in parser_args.meanstd_vars:
        temp_data = np.reshape(np.concatenate(stdDS[var].data, axis=0), [-1, n_pixels, 1])
        data_all.append(temp_data)
    dataset_std_v1 = np.concatenate(data_all, axis=2)

    data_all = []
    for c in range(int(len(dataset_out) / len(dataset_std_v1))):
        data_all.append(dataset_std_v1)

    dataset_std = np.concatenate(data_all, axis=0)

    dataset_in, dataset_out, dataset_mean, dataset_std = shuffle_data_meanstd(dataset_in, dataset_out, dataset_mean,
                                                                              dataset_std)

    '''if parser_args.time_lag > 0:
        dataset_in = dataset_in[:-parser_args.time_lag]
        dataset_out = dataset_out[parser_args.time_lag:]'''

    combined_data = np.concatenate((dataset_in, dataset_mean, dataset_std, dataset_out), axis=2)

    train_data, temp = train_test_split(combined_data, train_size=parser_args.partition[0], random_state=43)
    val_data, test_data = train_test_split(temp, test_size=parser_args.partition[2] / (
            parser_args.partition[1] + parser_args.partition[2]), random_state=43)

    dataloader_train = DataLoader(train_data, batch_size=parser_args.batch_size, shuffle=True, num_workers=12,
                                  collate_fn=sunet_collate)
    dataloader_validation = DataLoader(val_data, batch_size=parser_args.batch_size, shuffle=False, num_workers=12,
                                       collate_fn=sunet_collate)
    dataloader_test = DataLoader(test_data, batch_size=parser_args.batch_size, shuffle=False, num_workers=12,
                                 collate_fn=sunet_collate)
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

    #print(unet)
    # unet = unet.to(device)
    unet, device = init_device(parser_args.device, unet)
    lr = parser_args.learning_rate
    loss = torch.nn.MSELoss()
    optimizer = optim.Adam(unet.parameters(), lr=lr)

    '''trainer = create_supervised_trainer(unet, optimizer=optimizer, loss_fn=criterion, prepare_batch=custom_prepare_batch)'''

    def trainer(engine, batch):

        data_in, data_out = batch

        print(data_in)
        #data_in = data_in_initial.cpu().detach().numpy()
        #print("data in", data_in.shape)
        #print("input", data_in_initial.shape)
        # print("output", data_out_initial.shape)
        #data_in = data_in[:, :, :]
        # print("input revised", data_in.shape)
        # data_in, data_out, data_mean, data_std = batch
        # data_out = data_out_initial[:, :, 0:3]
        # data_mean = data_in_initial[:, :, 7:10]
        # data_std = data_in_initial[:, :, 10:]
        #data_in = torch.Tensor(data_in)
        data_in = data_in.to(device)
        #data_out = torch.Tensor(data_out)
        data_out = data_out.to(device)
        # data_mean = data_mean.to(device)
        # data_std = data_std.to(device)
        optimizer.zero_grad()
        #print(unet)
        unet.train()
        outputs = unet(data_in)


        # data_out_unscaled = (data_out * data_std) + data_mean
        # outputs_unscaled = (outputs * data_std) + data_mean

        # Precipitation constraint
        # outputs_unscaled[outputs_unscaled[:, :, 2] < 0] = 0

        # normalize
        # outputs = (outputs_unscaled - data_mean) / data_std

        # outputs = precip_pos(outputs_unscaled)
        # data_out = data_out[:,0:int(data_out.shape[1])-1, :] # removing the extra dimension of one_hot encoding
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

    #engine_train.add_event_handler(Events.EPOCH_STARTED, lambda x: print("Starting Epoch: {}".format(x.state.epoch)))

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

    pbar = ProgressBar()
    pbar.attach(engine_train, output_transform=lambda x: {"loss": x})
    engine_train.run(dataloader_train, max_epochs=parser_args.n_epochs)


    #trainer.run(dataloader_train, max_epochs=parser_args.n_epochs)

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


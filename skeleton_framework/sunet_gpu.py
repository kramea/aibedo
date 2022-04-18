from models.cnn import CNN
import numpy as np
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
    timelimit = batchShape[0] - 1

    data_in_array = np.array([item[:, 0:varlimit] for item in batch])
    data_out_array = np.array([item[timelimit:, varlimit:] for item in batch])

    data_in = torch.Tensor(data_in_array)
    data_out = torch.Tensor(data_out_array)
    return [data_in, data_out]



def get_dataloader(parser_args):
    glevel = int(parser_args.depth)
    n_pixels = icosahedron_nodes_calculator(glevel)
    lag = int(parser_args.time_lag)


    print("Grid level:", glevel)
    print("N pixels:", n_pixels)
    print("time lag:", lag)

    temp_folder = "/data/kramea/npy_files/"  # Change this to where you want .npy files are saved
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

    dataset = np.load(in_temp_npy_file)
    dataset_out = np.load(out_temp_npy_file)

    if parser_args.time_lag > 0:
        dataset = dataset[:-parser_args.time_lag]
        dataset_out = dataset_out[parser_args.time_lag:]

    combined_data = np.concatenate((dataset, dataset_out), axis=2)

    train_data, temp = train_test_split(combined_data, train_size=parser_args.partition[0], random_state=43)
    val_data, test_data = train_test_split(temp, test_size=parser_args.partition[2] / (
                parser_args.partition[1] + parser_args.partition[2]), random_state=43)

    dataloader_train = DataLoader(train_data, batch_size=parser_args.batch_size, shuffle=True, num_workers=12, collate_fn=sunet_collate)
    dataloader_validation = DataLoader(val_data, batch_size=parser_args.batch_size, shuffle=False, num_workers=12, collate_fn=sunet_collate)
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
            f"Training Results - Iteration: {engine_train.state.iteration}  Avg loss: {metrics['mse']:.2f}")

    @engine_train.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(dataloader_train)
        metrics = evaluator.state.metrics
        print(
            f"Training Results - Epoch: {engine_train.state.epoch}  Avg loss: {metrics['mse']:.2f}")


    @engine_train.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(dataloader_validation)
        metrics = evaluator.state.metrics
        print(
            f"Validation Results - Epoch: {engine_train.state.epoch} Avg loss: {metrics['mse']:.2f}")

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

    predictions = np.empty((parser_args.batch_size, len(parser_args.output_vars),n_pixels))
    groundtruth = np.empty((parser_args.batch_size, len(parser_args.output_vars),n_pixels))
    for batch in dataloader_test:
        data_in, data_out = batch
        preds = unet(data_in)
        test_loss = criterion(preds.float(), data_out)
        print("Test loss", test_loss)
        pred_numpy = preds.detach().cpu().numpy()
        predictions = np.concatenate((predictions, pred_numpy), axis=0)
        groundtruth = np.concatenate((groundtruth, data_out.detach().cpu().numpy()), axis=0)

    np.save("./saved_model_lag_" + str(parser_args.time_lag) + "/prediction_"+str(parser_args.n_epochs)+"_"+str(test_loss)+".npy", predictions)
    np.save("./saved_model_lag_" + str(parser_args.time_lag) + "/groundtruth_"+str(parser_args.n_epochs)+".npy", groundtruth)



if __name__ == "__main__":
    PARSER_ARGS = parse_config(create_parser())
    main(PARSER_ARGS)





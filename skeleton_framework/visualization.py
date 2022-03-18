import numpy as np
import cartopy.crs as ccrs
import xarray as xr
from interpolate import *
import os

def mse(a1, a2):
    se = np.square(np.subtract(a1,a2))
    mse = se.mean()
    return mse


def visualize_2d_all(gt, pr, longitude, latitude, title, export_path=None):
    """Visualize the data on a 2D map

    Args:
        x (numpy.array): numpy array with data the size of the longitude and latitude
        longitude (numpy.array): longitude coordinates
        latitude (numpy.array): latitude coordinates
        export_path (string): path and name for saving
    """

    fig = plt.figure(figsize=(70, 10))
    ax = fig.add_subplot(1, 3, 1, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines()

    plt.subplot(1, 3, 1)
    plt.scatter(longitude, latitude, s=20, c=gt, cmap=plt.get_cmap("RdYlBu_r"), alpha=1)
    plt.clim(0, 1)
    plt.xticks([-180,-90, 0,+90, 180])
    plt.yticks([-90,-45,0,45,90])
    plt.title("ground truth "+title)
    plt.colorbar(cmap=plt.get_cmap("RdYlBu_r"),fraction=0.046, pad=0.04)

    ax = fig.add_subplot(1, 3, 2, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines()
    
    plt.subplot(1, 3, 2)
    plt.scatter(longitude, latitude, s=20, c=pr, cmap=plt.get_cmap("RdYlBu_r"), alpha=1)
    plt.xticks([-180,-90, 0,+90, 180])
    plt.yticks([-90,-45,0,45,90])
    plt.clim(0, 1)
    plt.title("prediction "+title)
    plt.colorbar(cmap=plt.get_cmap("RdYlBu_r"),fraction=0.046, pad=0.04)
    
    ax = fig.add_subplot(1, 3, 3, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines()
    
    plt.subplot(1, 3, 3)
    plt.scatter(longitude, latitude, s=20, c=gt-pr, cmap=plt.get_cmap("RdYlBu_r"), alpha=1)
    plt.xticks([-180,-90, 0,+90, 180])
    plt.yticks([-90,-45,0,45,90])
    plt.clim(-1, 1)
    plt.title("error (gt-pr)"+title)
    plt.colorbar(cmap=plt.get_cmap("RdYlBu_r"),fraction=0.046, pad=0.04)
    if export_path:
        plt.savefig(export_path)
        plt.clf()
        plt.cla()
        plt.close()

    else:
        plt.show()


def visualization(path_gt, path_pre, path_fig):
    gt = np.load(path_gt)
    pr = np.load(path_pre)
    print(np.shape(gt), np.shape(pr))
    n, _,_ = np.shape(gt)
    ch = ["psl", "tas", "pr"]
    os.mkdir(path_fig)
    for c in range(len(ch)):
        error = mse(gt[:,:,c], pr[:,:,c])
        print(ch[c],error)
        for i in range(n):
            error = mse(gt[i,:,c], pr[i,:,c])
            visualize_2d_all(gt[i,:,c],pr[i,:,c], lon, lat, ch[0]+" "+str(i), export_path=path_fig+str(ch[c])+"_"+str(i)+"_"+str(error)+".png")

lon = np.load("./data/lon.npy")
lat = np.load("./data/lat.npy")
visualization("./results_sunet/groundtruth_35_tensor(0.0110).npy","./results_sunet/prediction_35_tensor(0.0110).npy", "./results_sunet/fig_all/")

visualization("./results_sunet/exp1_groundtruth_10_tensor(0.0118).npy","./results_sunet/exp1_prediction_10_tensor(0.0118).npy", "./results_sunet/fig_exp1/")

visualization("./results_sunet/exp2_groundtruth_20_tensor(0.0109).npy","./results_sunet/exp2_prediction_20_tensor(0.0109).npy", "./results_sunet/fig_exp2/")

visualization("./results_sunet/exp3_groundtruth_5_tensor(0.0161).npy","./results_sunet/exp3_prediction_5_tensor(0.0161).npy", "./results_sunet/fig_exp3/")


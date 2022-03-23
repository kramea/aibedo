import numpy as np
import cartopy.crs as ccrs
import xarray as xr
from interpolate import *
import os
from spherical_unet.utils.samplings import icosahedron_nodes_calculator

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



def calculate_interpolated_lon_lat(ncdf_path, glevel):
    ds= xr.open_dataset(ncdf_path)
    var = list(ds.data_vars)[0]
    da = np.asarray(ds[var])[0]
    lon_list = list(np.asarray(ds[var].lon))
    lat_list = list(np.asarray(ds[var].lat))
    lon, lat, interpolated_value = interpolate_SphereIcosahedral(glevel, da, lon_list, lat_list)
    return lon, lat


def visualization(ncdf_path, glevel, path_gt, path_pre, path_fig, output_channels):
    lon, lat= calculate_interpolated_lon_lat(ncdf_path, glevel)
    gt = np.load(path_gt)
    pr = np.load(path_pre)
    n, _,_ = np.shape(gt)
    ch = output_channels
    os.mkdir(path_fig)
    for c in range(len(ch)):
        error = mse(gt[:,:,c], pr[:,:,c])
        print("++++++++++++++++++++++++++++++++++++++")
        print("MSE of "+str(ch[c])+" is : "+str(error))
        print("++++++++++++++++++++++++++++++++++++++")
        for i in range(n):
            error = mse(gt[i,:,c], pr[i,:,c])
            visualize_2d_all(gt[i,:,c],pr[i,:,c], lon, lat, ch[0]+" "+str(i), export_path=path_fig+str(ch[c])+"_"+str(i)+"_"+str(error)+".png")






def main():
    # fill those parameters out manually! #
    glevel = 3
    output_channels  = ['tas_pre', 'pr_pre', 'psl_pre']
    ncdf_path = "/home/ubuntu/Exp7_CESM2_r1i1p1f1_historical_Input.nc"
    ground_truth_path =  "./saved_model/groundtruth_10_tensor(0.0114).npy"  
    prediction_path =  "./saved_model/prediction_10_tensor(0.0114).npy"
    path_to_figures = "./figures/" 
    print(ncdf_path, glevel, ground_truth_path, prediction_path, path_to_figures, output_channels)

    visualization(ncdf_path, glevel, ground_truth_path, prediction_path, path_to_figures, output_channels)


if __name__ == "__main__":
    main()

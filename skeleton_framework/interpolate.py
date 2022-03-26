import sys
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
import pygsp as pg


def xyz2lonlat(x,y,z, radius=6371.0e6):
    """From cartesian geocentric coordinates to 2D geographic coordinates."""
    """ lon: [-180, 180], lat:[-90, 90] """
    latitude = np.arcsin(z / radius)/np.pi*180
    longitude = np.arctan2(y, x)/np.pi*180
    return longitude, latitude 


#refer: https://www.omnicalculator.com/math/bilinear-interpolation#bilinear-interpolation-formula
def bilinear_interpolation(x1,x,x2,y1,y,y2,q11,q21,q12,q22):
    """
     |----|----|----|----|
     |    | x1 |  x | x2 |
     |-------------------|
     | y1 |q11 |    |q21 |
     |-------------------|
     | y  |    | p  |    |
     |-------------------|
     | y2 |q12 |    | q22|
     |-------------------|
    """
    if x1 == x2 and y1 != y2:
        r1 = q11
        r2 = q12
        p = r1*(y2-y)/(y2-y1)+r2*(y-y1)/(y2-y1)
    elif x1 != x2 and y1 == y2:
        p = q11*(x2-x)/(x2-x1)+q21*(x-x1)/(x2-x1)
    else:         
        r1 = q11*(x2-x)/(x2-x1)+q21*(x-x1)/(x2-x1)
        r2 = q12*(x2-x)/(x2-x1)+q22*(x-x1)/(x2-x1)
        p = r1*(y2-y)/(y2-y1)+r2*(y-y1)/(y2-y1)
    #print(x1,x,x2,":::",y2,y,y1,":::", q11,q12,q21,q22, p)
    return p



def resolution_SphereIcosahedral(lon,lat):
    lat_list = []
    for i in range(len(lon)):
        if lon[i] == 0.0:
            lat_list.append(lat[i])
    lat_list.sort()
    j=lat_list.index(0.0)
    assert((lat_list[j]-lat_list[j-1])==(lat_list[j+1]-lat_list[j]))
    return (lat_list[j]-lat_list[j-1])*111 # km resolution


def resolution_SphereHealpixl(lon,lat):
    lon_list = []
    for i in range(len(lat)):
        if lat[i] == 0.0:
            lon_list.append(lon[i])
    lon_list.sort()
    resolution = []
    for j in range(len(lon_list)-1):
        resolution.append(lon_list[j+1]-lon_list[j])
    return np.mean(resolution)*111 #km



#level-0(2**0=1): 12 vertices
#level-1(2**1=2): 42 vertices
#level-2(2**2=4): 162 vertices
#level-3(2**3=8): 642 vertices
#level-4(2**4=16): 2562 vertices
#level-5(2**5=32): 10242 vertices
#level-6(2**6=64): 40962 vertices
#level-7(2**7=128): 163842 vertices
#level-8(2**8=256): 655362 vertices
#level-9(2**9=512): 2621442 vertices
def interpolate_SphereIcosahedral(level, input_array,lon_list, lat_list):
    #(1) generate graph of SphereIcosahedral
    graph = pg.graphs.SphereIcosahedral(2**level)
    #(2) extract lon, lat coordinate
    vertices = graph.coords # xyz coordinates
    #print("number of vertices for level "+str(level)+" (2**"+str(level)+")is "+str(graph.n_vertices))
    lon = []
    lat = []
    radius = 1
    for tmp_xyz in vertices:
        tmp_lon, tmp_lat = xyz2lonlat(tmp_xyz[0],tmp_xyz[1],tmp_xyz[2], radius=radius)
        lon.append(tmp_lon)
        lat.append(tmp_lat)
    #(3)bilinear interpolate along with lon, lat order
    num_lat, num_lon = np.shape(input_array)
    div_lon = float((180.0*2)/float(num_lon))
    div_lat = float((90.0*2)/float(num_lat))
    interpolated_result = []
    #print(len(lon_list), len(lat_list), np.shape(input_array))#288 192 (192, 288)
    for i in range(len(lon)):
        lon_val, lat_val = lon[i], lat[i]  
        lon_idx = int((lon_val+180)/float(div_lon))  
        lat_idx = int((lat_val+90.0)/float(div_lat))
        if lat_idx >= num_lat -1 and lon_idx >= num_lon -1:
            lat_idx = num_lat -1
            lon_idx = num_lon -1
            interploated_value = input_array[lat_idx,lon_idx]
        elif lon_idx >=  num_lon-1 and lat_idx < num_lat-1: 
            lon_idx = num_lon -1
            interploated_value = bilinear_interpolation(lon_list[lon_idx],lon_val+180,lon_list[lon_idx],lat_list[lat_idx],lat_val,lat_list[lat_idx+1],input_array[ lat_idx+1, lon_idx],input_array[lat_idx+1, lon_idx],input_array[lat_idx, lon_idx],input_array[lat_idx,lon_idx])
        elif lat_idx >= num_lat -1 and lon_idx < num_lon - 1: 
            lat_idx = num_lat - 1
            interploated_value = bilinear_interpolation(lon_list[lon_idx],lon_val+180,lon_list[lon_idx+1],lat_list[lat_idx],lat_val,lat_list[lat_idx],input_array[ lat_idx, lon_idx],input_array[lat_idx, lon_idx+1],input_array[lat_idx, lon_idx],input_array[lat_idx,lon_idx+1])
        elif lat_idx >= num_lat -1 and lon_idx >= num_lon -1:
            lat_idx = num_lat -1
            lon_idx = num_lon -1
            interploated_value = input_array[lat_idx,lon_idx]
        elif lat_idx < num_lat -1 and lon_idx < num_lon -1:
            interploated_value = bilinear_interpolation(lon_list[lon_idx],lon_val+180,lon_list[lon_idx+1],lat_list[lat_idx],lat_val,lat_list[lat_idx+1],input_array[ lat_idx+1, lon_idx],input_array[lat_idx+1, lon_idx+1],input_array[lat_idx, lon_idx],input_array[lat_idx,lon_idx+1])
        else:
            print("ERROR", lat_idx, lon_idx)
            exit()
        interpolated_result.append(interploated_value)
    # Kalai added Longitudinal shift 
    lon2 = [x+180 for x in lon]
    return lon2, lat, interpolated_result



#subdivision: 4 neighbors: 8 192 vertices
#subdivision: 8 neighbors: 8 768 vertices
#subdivision: 16 neighbors: 8 3072 vertices
#subdivision: 32 neighbors: 8 12288 vertices
#subdivision: 64 neighbors: 8 49152 vertices
#input_array should be 2d  array (lon, lat)
def interpolate_SphereHealpix(subdivision, neighbors, input_array, lon_list, lat_list):
    #(1) generate graph of SphereHealpix
    graph = pg.graphs.SphereHealpix(subdivision, k=neighbors)
    #(2) extract lon, lat coordinate
    vertices = graph.coords # xyz coordinates
    #print("number of vertices for sub-divsion "+str(subdivision)+" neighbor"+str(neighbors)+" is "+str(graph.n_vertices))
    lon = []
    lat = []
    radius = 1
    for tmp_xyz in vertices:
        tmp_lon, tmp_lat = xyz2lonlat(tmp_xyz[0],tmp_xyz[1],tmp_xyz[2], radius=radius)
        lon.append(tmp_lon)
        lat.append(tmp_lat)
    #(3)bilinear interpolate along with lon, lat order
    num_lat, num_lon = np.shape(input_array)
    div_lon = float((180.0*2)/float(num_lon))
    div_lat = float((90.0*2)/float(num_lat))
    interpolated_result = []
    #print(len(lon_list), len(lat_list), np.shape(input_array))#288 192 (192, 288)
    for i in range(len(lon)):
        lon_val, lat_val = lon[i], lat[i]
        lon_idx = int((lon_val+180)/float(div_lon))
        lat_idx = int((lat_val+90.0)/float(div_lat))
        if lat_idx >= num_lat -1 and lon_idx >= num_lon -1:
            lat_idx = num_lat -1
            lon_idx = num_lon -1
            interploated_value = input_array[lat_idx,lon_idx]
        elif lon_idx >=  num_lon-1 and lat_idx < num_lat-1:
            lon_idx = num_lon -1
            interploated_value = bilinear_interpolation(lon_list[lon_idx],lon_val+180,lon_list[lon_idx],lat_list[lat_idx],lat_val,lat_list[lat_idx+1],input_array[ lat_idx+1, lon_idx],input_array[lat_idx+1, lon_idx],input_array[lat_idx, lon_idx],input_array[lat_idx,lon_idx])
        elif lat_idx >= num_lat -1 and lon_idx < num_lon - 1:
            lat_idx = num_lat - 1
            interploated_value = bilinear_interpolation(lon_list[lon_idx],lon_val+180,lon_list[lon_idx+1],lat_list[lat_idx],lat_val,lat_list[lat_idx],input_array[ lat_idx, lon_idx],input_array[lat_idx, lon_idx+1],input_array[lat_idx, lon_idx],input_array[lat_idx,lon_idx+1])
        elif lat_idx >= num_lat -1 and lon_idx >= num_lon -1:
            lat_idx = num_lat -1
            lon_idx = num_lon -1
            interploated_value = input_array[lat_idx,lon_idx]
        elif lat_idx < num_lat -1 and lon_idx < num_lon -1:
            interploated_value = bilinear_interpolation(lon_list[lon_idx],lon_val+180,lon_list[lon_idx+1],lat_list[lat_idx],lat_val,lat_list[lat_idx+1],input_array[ lat_idx+1, lon_idx],input_array[lat_idx+1, lon_idx+1],input_array[lat_idx, lon_idx],input_array[lat_idx,lon_idx+1])
        else:
            print("ERROR", lat_idx, lon_idx)
            exit()
        interpolated_result.append(interploated_value)
    # Kalai added Longitudinal shift 
    lon2 = [x+180 for x in lon]
    return lon2, lat, interpolated_result




'''
import xarray as xr
path = "/Users/sookim/Desktop/ALBEDO/aibedo/scripts/data/ours/rsut_Amon_CESM2_historical_r1i1p1f1_gn_185001-201412.nc"
ds = xr.open_dataset(path)
data = np.asarray(ds.rsut[0])
lon_list = list(np.asarray(ds.rsut[0].lon))
lat_list = list(np.asarray(ds.rsut[0].lat))
lon, lat, interpolated_value = interpolate_SphereIcosahedral(8, data, lon_list, lat_list)
lon, lat, interpolated_value = interpolate_SphereHealpix(4, 8, data, lon_list, lat_list)
print("done")
'''


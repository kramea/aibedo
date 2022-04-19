import pygsp as pg
import numpy as np

### Generates a text file with centre points of a icosahedral sphere grid
### Formatted to pass as an argument for 'cdo remap,GRIDFILE in.nc out.nc'

def xyz2lonlat(x:float,y:float,zLfloat, radius=6371.0e6):
    """From cartesian geocentric coordinates to 2D geographic coordinates."""
    """ lon: [-180, 180], lat:[-90, 90] """
    latitude = np.arcsin(z / radius)/np.pi*180
    longitude = np.arctan2(y, x)/np.pi*180
    return longitude, latitude 

def gen_icosphere(level:int,radius:float = 6371.0e6):
    '''
    Generates points on a icosphere grid (from Soo Kim)
    level (int): iteration level
    radius (float): radius of the earth
    '''
    #(1) generate graph of SphereIcosahedral
    graph = pg.graphs.SphereIcosahedral(2**level)
    #(2) extract lon, lat coordinate
    vertices = graph.coords # xyz coordinates
    #print("number of vertices for level "+str(level)+" (2**"+str(level)+")is "+str(graph.n_vertices))
    lon = []
    lat = []
    radius = 1
    # convert cartesian points to spherical lon/lat
    for tmp_xyz in vertices:
        tmp_lon, tmp_lat = xyz2lonlat(tmp_xyz[0],tmp_xyz[1],tmp_xyz[2], radius=radius)
        lon.append(tmp_lon)
        lat.append(tmp_lat)
    return lon, lat

sphlevel = 6

lon,lat = gen_icosphere(sphlevel,radius = 6371.0e6)

f = open("isosphere_{0}.txt".format(sphlevel), "a")

# write grid to file
f.write("gridtype = unstructured\n")
f.write("gridsize = {0}\n".format(len(lon)))
f.write("# Longitudes\n")
f.write("xvals = " + ' '.join(map(str, lon))+"\n")
f.write("# Latitudes\n")
f.write("yvals = " + ' '.join(map(str, lat))+"\n")

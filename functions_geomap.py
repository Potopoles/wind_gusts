import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset

inp_const_path = '../data/constants_c1.nc'

def draw_map(line_res, grid_res):

    # load topography
    nco = Dataset(inp_const_path, 'r')
    hsurf = nco['HSURF'][:]
    lon = nco['lon_1'][:]
    lat = nco['lat_1'][:]

    fig = plt.figure(figsize=(15,9))
    m = Basemap(projection='lcc', resolution=line_res,
                lon_0=8.5, width=0.65E6,
                lat_0=47.1, height=0.4E6)

    m.pcolormesh(lon,lat,hsurf, latlon=True, cmap='terrain', vmin=-1000,vmax=4200, zorder=-1)

    m.drawcountries(linewidth=1.0)
    m.drawcoastlines(linewidth=1.0, color='0.2')
    m.drawlsmask(land_color=(1,1,1,0.), ocean_color=(0,0.3,1,0.5),
                zorder=1, resolution=line_res, grid=grid_res)
    m.drawrivers(linewidth=1.0, color=(0,0.3,1,0.2))
    return(m)

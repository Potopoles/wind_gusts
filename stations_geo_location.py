import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from namelist_cases import Case_Namelist
from functions_geomap import draw_map

i_plot = 1
inp_plot_name = 'stations_geo_location.png'
# resolution of spatial lines (c,l,i,h,f)
inp_line_res = 'h'
#inp_line_res = 'c'
# resolution of spatial areas (10,5,2.5,1.25)
inp_grid_res = 1.25
#inp_grid_res = 10
inp_marker = '.'


station_meta = pd.read_csv(Case_Namelist.stations_meta_path, encoding='ISO-8859-1',
                            error_bad_lines=False, sep=';')
stations_meta_use = station_meta[station_meta['Use'] == 'y']
points_lon = stations_meta_use['Lon'].values
points_lat = stations_meta_use['Lat'].values


# draw plot
m = draw_map(inp_line_res, inp_grid_res)
m.scatter(points_lon, points_lat, zorder=2, color='black', latlon=True,
            marker=inp_marker)


if i_plot == 1:
    plt.show()
elif i_plot == 2:
    plt.savefig(Case_Namelist.plot_base_dir + '/' + inp_plot_name)

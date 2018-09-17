import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
#from functions import plot_error
import globals as G
from namelist_cases import Case_Namelist
from functions_geomap import draw_map

############ USER INPUT #############
train_case_index = 10
CNtrain = Case_Namelist(train_case_index)
apply_case_index = 0
CNapply = Case_Namelist(apply_case_index)
# do not plot (0) show plot (1) save plot (2)
i_plot = 1
model_dt = 10
i_label = ''
# resolution of spatial lines (c,l,i,h,f)
inp_line_res = 'h'
inp_line_res = 'c'
# resolution of spatial areas (10,5,2.5,1.25)
inp_grid_res = 1.25
inp_grid_res = 10
inp_marker = '.'
#####################################

# create directories
if i_plot > 1 and not os.path.exists(CNapply.plot_path):
    os.mkdir(CNapply.plot_path)

# LOAD PARAMS
params = pickle.load( open(CNtrain.params_readj_path, 'rb') )

# LAOD DATA
data = pickle.load( open(CNapply.train_readj_path, 'rb') )
model_mean = data['model_mean']
obs_gust = data['obs_gust']
obs_mean = data['obs_mean']
tcm = data['tcm']
zvp10 = data['zvp10']
stations = data['stations']
print(stations)


station_meta = pd.read_csv(Case_Namelist.stations_meta_path, encoding='ISO-8859-1',
                            error_bad_lines=False, sep=';')
df_stat = station_meta[station_meta['ABBR'].isin(stations)]
points_lon = []
points_lat = []
for stat in stations:
    selection = df_stat[df_stat['ABBR'] == stat]
    points_lon.extend(selection['Lon'].values)
    points_lat.extend(selection['Lat'].values)
print(points_lon)
print(points_lat)

# obs nan mask
obsmask = np.isnan(obs_gust)
obsmask[np.isnan(obs_mean)] = True


obs_gust[obsmask] = np.nan
obs_mean[obsmask] = np.nan
model_mean[obsmask] = np.nan
tcm[obsmask] = np.nan
zvp10[obsmask] = np.nan
print(zvp10.shape)
print(model_mean.shape)

m = draw_map(inp_line_res, inp_grid_res)
m.scatter(points_lon, points_lat, zorder=2, color='black', latlon=True,
            marker=inp_marker)


if i_plot == 1:
    plt.show()
elif i_plot == 2:
    plt.savefig(Case_Namelist.plot_base_dir + '/' + inp_plot_name)
quit()


for mode in params.keys():
    print('#################################################################################')
    print('############################## ' + str(mode) + ' ################################')

    alphas = params[mode]['alphas']
    print(alphas)

    # Calculate final gust
    # Calculate final gust
    gust = zvp10 + alphas['1']*tcm*zvp10 + alphas['2']*tcm*zvp10**2

    maxid = gust.argmax(axis=2)
    I,J = np.indices(maxid.shape)
    gust_max = gust[I,J,maxid].squeeze()
    print(gust_max.shape)
    quit()

    if mode == 'ln':
        gust_orig = zvp10 + 7.2*tcm*zvp10
    elif mode == 'nl':
        gust_orig = zvp10 + 7.2*tcm*zvp10 + 0.09*tcm*zvp10**2
    gust_max_orig = np.amax(gust_orig,axis=1)


    plot_error(obs_gust, model_mean, obs_mean, gust_max, gust_max_orig)
    plt.suptitle('apply READJUST  '+mode)

    if i_plot == 1:
        plt.show()
    elif i_plot > 1:
        if i_label == '':
            plot_name = CNapply.plot_path + 'applied_readj_sw_'+i_sample_weight+'_'+str(mode)+'.png'
        else:
            plot_name = CNappyl.plot_path + 'applied_readj_sw_'+i_sample_weight+'_'+str(i_label)+'_'+str(mode)+'.png'
        print(plot_name)
        plt.savefig(plot_name)
        plt.close('all')


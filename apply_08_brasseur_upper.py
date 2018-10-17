import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from functions import plot_error
import globals as G
from namelist_cases import Case_Namelist
import namelist_cases as nl
from functions_train import braes_feature_matrix, braes_feature_matrix_timestep

############ USER INPUT #############
train_case_index = nl.train_case_index
apply_case_index = nl.apply_case_index
CNtrain = Case_Namelist(train_case_index)
CNapply = Case_Namelist(apply_case_index)
# do not plot (0) show plot (1) save plot (2)
i_plot = nl.apply_i_plot
model_dt = nl.apply_model_dt
i_label = ''
apply_on_hourly_gusts = 0
#####################################

# create directories
if i_plot > 1 and not os.path.exists(CNapply.plot_path):
    os.mkdir(CNapply.plot_path)

# LOAD PARAMS
params = pickle.load( open(CNtrain.params_braub_path, 'rb') )
#print(params)
#quit()

# LAOD DATA
data = pickle.load( open(CNapply.train_braub_path, 'rb') )
model_mean = data['model_mean']
gust_bra = data['gust_bra']
kheight = data['kheight']
height = data['height']
obs_gust = data['obs_gust']
obs_mean = data['obs_mean']


######################################################
################ FINAL TEST
#from functions import load_var_at_station_from_nc
#test_what = 'gust_mean'
#test_what = 'braest'
#if test_what == 'gust_mean':
#    nc_path = '/scratch/heimc/tuned_gusts_pompa/itype_diag_gust_5/VMAX_10M.nc'
#elif test_what == 'braest':
#    nc_path = '/scratch/heimc/tuned_gusts_pompa/itype_diag_gust_3/VMAX_10M.nc'
#var_name = 'VMAX_10M'
#sel_stat = 'ABO'
#sel_stat = 'KLO'
#gust_nc = load_var_at_station_from_nc(nc_path, var_name, sel_stat)
#df = pd.DataFrame(gust_nc, columns=['nc'])
#data = pickle.load( open(CNapply.mod_path, 'rb') )
#stat_keys = data[G.STAT_NAMES]
#stat_ind = np.argwhere(np.asarray(stat_keys) == sel_stat)[0][0]
#if test_what == 'gust_mean':
#    mode = 'gust_mean'
#    alphas = params[mode]
#    gust = alphas[0] + alphas[1]*gust_bra + alphas[2]*model_mean
#elif test_what == 'braest':
#    gust = gust_bra
#gust_pyth = np.max(gust[:,stat_ind,:], axis=1)
#df['python'] = gust_pyth
#diff = gust_pyth - gust_nc
#df['diff'] = diff
#df['relDiff'] = diff/gust_pyth
#print(df)
#print('max error ' + str(np.max(df['diff'])))
#quit()
######################################################


# observation to 1D and filter values
obsmask = np.isnan(obs_gust)
obsmask[np.isnan(obs_mean)] = True
model_mean_hr = np.mean(model_mean, axis=2)
obs_gust = obs_gust[~obsmask] 
obs_mean = obs_mean[~obsmask] 
model_mean = model_mean[~obsmask]
model_mean_hr = model_mean_hr[~obsmask]
gust_bra = gust_bra[~obsmask]
kheight = kheight[~obsmask]
height = height[~obsmask]

N = obs_gust.flatten().shape[0]
print(N)




# find maximum gust
if apply_on_hourly_gusts:
    maxid = gust_bra.argmax(axis=1)
    I = np.indices(maxid.shape)
    gust_bra_max = gust_bra[I,maxid].flatten()
    gust_bra_max_unscaled = gust_bra[I,maxid].flatten()
    model_mean_max = model_mean[I,maxid].flatten() 
    gust_bra_max = gust_bra[I,maxid].flatten()
    kheight_max = kheight[I,maxid].flatten()
    height_max = height[I,maxid].flatten()
else:
    maxid = gust_bra.argmax(axis=1)
    I = np.indices(maxid.shape)
    gust_bra_max_unscaled = gust_bra[I,maxid].flatten()

for mode in params.keys():
    print('#################################################################################')
    print('############################## ' + str(mode) + ' ################################')

    alphas = params[mode]

    if apply_on_hourly_gusts:
        X = braes_feature_matrix(mode, gust_bra_max, kheight_max,
                                        height_max, model_mean_max)
        gust_max = np.sum(X*alphas, axis=1)
    else:
        X = braes_feature_matrix_timestep(mode, gust_bra, kheight,
                                        height, model_mean)
        gust = np.sum(X*alphas, axis=2)
        gust_max = np.max(gust,axis=1)


    if i_plot > 0:
        plot_error(obs_gust, model_mean_hr, obs_mean, gust_max, gust_bra_max_unscaled)
        plt.suptitle('apply BRAUB  '+mode)

        if i_plot == 1:
            plt.show()
        elif i_plot > 1:
            if i_label == '':
                plot_name = CNapply.plot_path + 'applied_braub_'+str(mode)+'.png'
            else:
                plot_name = CNappyl.plot_path + 'applied_braub_'+str(i_label)+'_'+str(mode)+'.png'
            print(plot_name)
            plt.savefig(plot_name)
            plt.close('all')


import numpy as np
import os
import copy
import matplotlib.pyplot as plt
import pickle
from functions import plot_error
import globals as G
from namelist_cases import Case_Namelist
import namelist_cases as nl
from functions_train import icon_feature_matrix, icon_feature_matrix_timestep

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
params = pickle.load( open(CNtrain.params_icon_path, 'rb') )
#print(params)
#quit()

data = pickle.load( open(CNapply.train_icon_path, 'rb') )
model_mean = data['model_mean']
model_mean = data['model_mean']
gust_ico = data['gust_ico']
tkel1 = data['tkel1']
dvl3v10 = data['dvl3v10']
height = data['height']
obs_gust = data['obs_gust']
obs_mean = data['obs_mean']

obsmask = np.isnan(obs_gust)
obsmask[np.isnan(obs_mean)] = True
model_mean_hr = np.mean(model_mean, axis=2)
obs_gust = obs_gust[~obsmask] 
obs_mean = obs_mean[~obsmask] 
model_mean = model_mean[~obsmask]
model_mean_hr = model_mean_hr[~obsmask]
gust_ico = gust_ico[~obsmask]
tkel1 = tkel1[~obsmask]
dvl3v10 = dvl3v10[~obsmask]
height = height[~obsmask]
N = obs_gust.flatten().shape[0]

# find maximum gust

if apply_on_hourly_gusts:
    maxid = gust_ico.argmax(axis=1)
    I = np.indices(maxid.shape)
    model_mean_max = model_mean[I,maxid].flatten() 
    gust_ico_max = gust_ico[I,maxid].flatten()
    gust_ico_max_unscaled = gust_ico[I,maxid].flatten()
    tkel1_max = tkel1[I,maxid].flatten()
    dvl3v10_max = dvl3v10[I,maxid].flatten()
    height_max = height[I,maxid].flatten()
else:
    maxid = gust_ico.argmax(axis=1)
    I = np.indices(maxid.shape)
    gust_ico_max_unscaled = gust_ico[I,maxid].flatten()

for mode in params.keys():
    print('#################################################################################')
    print('############################## ' + str(mode) + ' ################################')

    alphas = params[mode]

    if apply_on_hourly_gusts:
        X = icon_feature_matrix(mode, gust_ico_max, height_max,
                                    dvl3v10_max, model_mean_max,
                                    tkel1_max)
        gust_max = np.sum(X*alphas, axis=1)
    else:
        X = icon_feature_matrix_timestep(mode, gust_ico, height,
                                    dvl3v10, model_mean,
                                    tkel1)
        gust = np.sum(X*alphas, axis=2)
        gust_max = np.max(gust,axis=1)


    plot_error(obs_gust, model_mean_hr, obs_mean, gust_max, gust_ico_max_unscaled)
    plt.suptitle('ICON  '+mode)

    if i_plot == 1:
        plt.show()
    elif i_plot > 1:
        if i_label == '':
            plot_name = CNapply.plot_path + 'applied_icon_'+str(mode)+'.png'
        else:
            plot_name = CNapply.plot_path + 'applied_icon_'+str(i_label)+'_'+str(mode)+'.png'
        print(plot_name)
        plt.savefig(plot_name)
        plt.close('all')


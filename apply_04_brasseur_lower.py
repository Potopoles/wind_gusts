import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import pickle
from functions import plot_error
import globals as G
from namelist_cases import Case_Namelist
import namelist_cases as nl
from functions_train import bralb_feature_matrix, bralb_feature_matrix_timestep

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
params = pickle.load( open(CNtrain.params_bralb_path, 'rb') )
#print(params)
#quit()

# LAOD DATA
data = pickle.load( open(CNapply.train_bralb_path, 'rb') )
model_mean = data['model_mean']
gust_lb = data['gust_lb']
kheight_lb = data['kheight_lb']
height = data['height']
obs_gust = data['obs_gust']
obs_mean = data['obs_mean']

# observation to 1D and filter values
obsmask = np.isnan(obs_gust)
obsmask[np.isnan(obs_mean)] = True
model_mean_hr = np.mean(model_mean, axis=2)
obs_gust = obs_gust[~obsmask] 
obs_mean = obs_mean[~obsmask] 
model_mean = model_mean[~obsmask]
model_mean_hr = model_mean_hr[~obsmask]
gust_lb = gust_lb[~obsmask]
kheight_lb = kheight_lb[~obsmask]
height = height[~obsmask]

# find maximum gust
if apply_on_hourly_gusts:
    maxid = gust_lb.argmax(axis=1)
    I = np.indices(maxid.shape)
    gust_lb_max = gust_lb[I,maxid].flatten()
    gust_lb_max_unscaled = gust_lb[I,maxid].flatten()
    model_mean_max = model_mean[I,maxid].flatten() 
    gust_lb_max = gust_lb[I,maxid].flatten()
    kheight_lb_max = kheight_lb[I,maxid].flatten()
    height_max = height[I,maxid].flatten()
else:
    maxid = gust_lb.argmax(axis=1)
    I = np.indices(maxid.shape)
    gust_lb_max_unscaled = gust_lb[I,maxid].flatten()

for mode in params.keys():
    print('#################################################################################')
    print('############################## ' + str(mode) + ' ################################')

    alphas = params[mode]

    if apply_on_hourly_gusts:
        X = bralb_feature_matrix(mode, gust_lb_max, kheight_lb_max,
                                        height_max, model_mean_max)
        gust_max = np.sum(X*alphas, axis=1)
    else:
        X = bralb_feature_matrix_timestep(mode, gust_lb, kheight_lb,
                                        height, model_mean)
        gust = np.sum(X*alphas, axis=2)
        gust_max = np.max(gust,axis=1)

    if i_plot > 0:
        plot_error(obs_gust, model_mean_hr, obs_mean, gust_max, gust_lb_max_unscaled)
        plt.suptitle('apply BRALB  '+mode)

        if i_plot == 1:
            plt.show()
        elif i_plot > 1:
            if i_label == '':
                plot_name = CNapply.plot_path + 'applied_bralb_'+str(mode)+'.png'
            else:
                plot_name = CNappyl.plot_path + 'applied_bralb_'+str(i_label)+'_'+str(mode)+'.png'
            print(plot_name)
            plt.savefig(plot_name)
            plt.close('all')



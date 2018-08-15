import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import pickle
from functions import plot_error
import globals as G
from namelist_cases import Case_Namelist
from functions_train import braes_feature_matrix, braes_feature_matrix_timestep

############ USER INPUT #############
train_case_index = 10
CNtrain = Case_Namelist(train_case_index)
apply_case_index = 10
CNapply = Case_Namelist(apply_case_index)
# do not plot (0) show plot (1) save plot (2)
i_plot = 2
model_dt = 10
i_label = ''

i_sample_weight = '1'
apply_on_hourly_gusts = 0
#####################################

# create directories
if i_plot > 1 and not os.path.exists(CNapply.plot_path):
    os.mkdir(CNapply.plot_path)

# LOAD PARAMS
params = pickle.load( open(CNtrain.params_braes_path, 'rb') )
#print(params)
#quit()

# LAOD DATA
data = pickle.load( open(CNapply.train_braes_path, 'rb') )
model_mean = data['model_mean']
gust_est = data['gust_est']
kheight_est = data['kheight_est']
height = data['height']
obs_gust = data['obs_gust']
obs_mean = data['obs_mean']

# observation to 1D and filter values
obsmask = np.isnan(obs_gust)
model_mean_hr = np.mean(model_mean, axis=2)
obs_gust = obs_gust[~obsmask] 
obs_mean = obs_mean[~obsmask] 
model_mean = model_mean[~obsmask]
model_mean_hr = model_mean_hr[~obsmask]
gust_est = gust_est[~obsmask]
kheight_est = kheight_est[~obsmask]
height = height[~obsmask]

# find maximum gust
if apply_on_hourly_gusts:
    maxid = gust_est.argmax(axis=1)
    I = np.indices(maxid.shape)
    gust_est_max = gust_est[I,maxid].flatten()
    gust_est_max_unscaled = gust_est[I,maxid].flatten()
    model_mean_max = model_mean[I,maxid].flatten() 
    gust_est_max = gust_est[I,maxid].flatten()
    kheight_est_max = kheight_est[I,maxid].flatten()
    height_max = height[I,maxid].flatten()
else:
    maxid = gust_est.argmax(axis=1)
    I = np.indices(maxid.shape)
    gust_est_max_unscaled = gust_est[I,maxid].flatten()

for mode in params.keys():
    print('#################################################################################')
    print('############################## ' + str(mode) + ' ################################')

    alphas = params[mode]

    if apply_on_hourly_gusts:
        X = braes_feature_matrix(mode, gust_est_max, kheight_est_max,
                                        height_max, model_mean_max)
        gust_max = np.sum(X*alphas, axis=1)
    else:
        X = braes_feature_matrix_timestep(mode, gust_est, kheight_est,
                                        height, model_mean)
        gust = np.sum(X*alphas, axis=2)
        gust_max = np.max(gust,axis=1)

    plot_error(obs_gust, model_mean_hr, obs_mean, gust_max, gust_est_max_unscaled)
    plt.suptitle('apply BRAEST  '+mode)

    if i_plot == 1:
        plt.show()
    elif i_plot > 1:
        if i_label == '':
            plot_name = CNapply.plot_path + 'applied_braes_sw_'+i_sample_weight+'_'+str(mode)+'.png'
        else:
            plot_name = CNappyl.plot_path + 'applied_braes_sw_'+i_sample_weight+'_'+str(i_label)+'_'+str(mode)+'.png'
        print(plot_name)
        plt.savefig(plot_name)
        plt.close('all')



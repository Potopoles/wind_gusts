import numpy as np
import os
import copy
import matplotlib.pyplot as plt
import pickle
from functions import plot_error
import globals as G
from namelist_cases import Case_Namelist
from functions_train import icon_feature_matrix

############ USER INPUT #############
train_case_index = 0
CNtrain = Case_Namelist(train_case_index)
apply_case_index = 0
CNapply = Case_Namelist(apply_case_index)
# do not plot (0) show plot (1) save plot (2)
i_plot = 2
i_scaling = 1
i_label = ''
max_mean_wind_error = 1.0
#i_sample_weight = 'linear'
#i_sample_weight = 'squared'
i_sample_weight = '1'
#####################################

# create directories
if i_plot > 1 and not os.path.exists(CNapply.plot_path):
    os.mkdir(CNapply.plot_path)

# LOAD PARAMS
params = pickle.load( open(CNtrain.params_icon_path, 'rb') )

data = pickle.load( open(CNapply.train_icon_path, 'rb') )
model_mean_max = data['model_mean_max']
model_mean = data['model_mean']
gust_ico_max = data['gust_ico_max']
tkel1_max = data['tkel1_max']
dvl3v10_max = data['dvl3v10_max']
height_max = data['height_max']
obs_gust_flat = data['obs_gust_flat']
obs_mean_flat = data['obs_mean_flat']
gust_ico_max_unscaled = data['gust_ico_max_unscaled']

mean_abs_error = np.abs(model_mean - obs_mean_flat)
mean_rel_error = mean_abs_error/obs_mean_flat
errormask = mean_rel_error > max_mean_wind_error

model_mean_max = model_mean_max[~errormask]
model_mean = model_mean[~errormask]
gust_ico_max = gust_ico_max[~errormask]
tkel1_max = tkel1_max[~errormask]
dvl3v10_max = dvl3v10_max[~errormask]
height_max = height_max[~errormask]
obs_gust_flat = obs_gust_flat[~errormask]
obs_mean_flat = obs_mean_flat[~errormask]
gust_ico_max_unscaled = gust_ico_max_unscaled[~errormask]

for mode in params.keys():
    print('#################################################################################')
    print('############################## ' + str(mode) + ' ################################')

    alphas = params[mode]

    # calc current time step gusts
    X = icon_feature_matrix(mode, gust_ico_max, height_max,
                                dvl3v10_max, model_mean_max,
                                tkel1_max)
    # Calculate final gust
    gust_max = np.sum(X*alphas, axis=1)

    plot_error(obs_gust_flat, model_mean, obs_mean_flat, gust_max, gust_ico_max_unscaled)
    plt.suptitle('ICON  '+mode)

    if i_plot == 1:
        plt.show()
    elif i_plot > 1:
        if i_label == '':
            plot_name = CNapply.plot_path + 'tuning_icon_sw_'+i_sample_weight+'_mwa_'+str(max_mean_wind_error)+'_'\
                                        +str(mode)+'.png'
        else:
            plot_name = CNapply.plot_path + 'tuning_icon_sw_'+i_sample_weight+'_mwa_'+str(max_mean_wind_error)+'_'\
                                        +str(i_label)+'_'+str(mode)+'.png'
        print(plot_name)
        plt.savefig(plot_name)
        plt.close('all')


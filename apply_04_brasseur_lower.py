import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import pickle
from functions import plot_error
import globals as G
from namelist_cases import Case_Namelist
from functions_train import bralb_feature_matrix

############ USER INPUT #############
train_case_index = 10
CNtrain = Case_Namelist(train_case_index)
apply_case_index = 12
CNapply = Case_Namelist(apply_case_index)
# do not plot (0) show plot (1) save plot (2)
i_plot = 2
model_dt = 10
i_label = ''

max_mean_wind_error = 1.0
i_sample_weight = '1'
#####################################

# create directories
if i_plot > 1 and not os.path.exists(CNapply.plot_path):
    os.mkdir(CNapply.plot_path)

# LOAD PARAMS
params = pickle.load( open(CNtrain.params_bralb_path, 'rb') )

# LAOD DATA
data = pickle.load( open(CNapply.train_bralb_path, 'rb') )
model_mean_max = data['model_mean_max']
model_mean = data['model_mean']
gust_lb_max = data['gust_lb_max']
kheight_lb_max = data['kheight_lb_max']
height_max = data['height_max']
obs_gust_flat = data['obs_gust_flat']
obs_mean_flat = data['obs_mean_flat']
gust_lb_max_original = data['gust_lb_max_unscaled']


mean_abs_error = np.abs(model_mean - obs_mean_flat)
mean_rel_error = mean_abs_error/obs_mean_flat
errormask = mean_rel_error > max_mean_wind_error

model_mean_max = model_mean_max[~errormask]
model_mean = model_mean[~errormask]
gust_lb_max = gust_lb_max[~errormask]
kheight_lb_max = kheight_lb_max[~errormask]
height_max = height_max[~errormask]
obs_gust_flat = obs_gust_flat[~errormask]
obs_mean_flat = obs_mean_flat[~errormask]
gust_lb_max_original = gust_lb_max_original[~errormask]

for mode in params.keys():
    print('#################################################################################')
    print('############################## ' + str(mode) + ' ################################')

    alphas = params[mode]

    # calc current time step gusts
    X = bralb_feature_matrix(mode, gust_lb_max, kheight_lb_max,
                                    height_max, model_mean_max)

    # Calculate final gust
    gust_max = np.sum(X*alphas, axis=1)

    plot_error(obs_gust_flat, model_mean, obs_mean_flat, gust_max, gust_lb_max_original)
    plt.suptitle('apply BRAEST  '+mode)

    if i_plot == 1:
        plt.show()
    elif i_plot > 1:
        if i_label == '':
            plot_name = CNapply.plot_path + 'applied_bralb_sw_'+i_sample_weight+'_'+str(mode)+'.png'
        else:
            plot_name = CNappyl.plot_path + 'applied_bralb_sw_'+i_sample_weight+'_'+str(i_label)+'_'+str(mode)+'.png'
        print(plot_name)
        plt.savefig(plot_name)
        plt.close('all')



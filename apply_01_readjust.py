import os
import copy
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import pickle
from functions import plot_error
import globals as G
from namelist_cases import Case_Namelist

############ USER INPUT #############
train_case_index = 10
CNtrain = Case_Namelist(train_case_index)
apply_case_index = 10
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
params = pickle.load( open(CNtrain.params_readj_path, 'rb') )

# LAOD DATA
data = pickle.load( open(CNapply.train_readj_path, 'rb') )
model_mean = data['model_mean']
obs_gust = data['obs_gust']
obs_mean = data['obs_mean']
tcm = data['tcm']
zvp10 = data['zvp10']


# obs nan mask
obsmask = np.isnan(obs_gust)
obsmask[np.isnan(obs_mean)] = True
# bad mean wind accuracy mask
mean_abs_error = np.abs(model_mean - obs_mean)
mean_rel_error = mean_abs_error/obs_mean
errormask = mean_rel_error > max_mean_wind_error
# combine both
obsmask[errormask] = True

obs_gust = obs_gust[~obsmask] 
obs_mean = obs_mean[~obsmask]
model_mean = model_mean[~obsmask]
tcm = tcm[~obsmask]
zvp10 = zvp10[~obsmask]


for mode in params.keys():
    print('#################################################################################')
    print('############################## ' + str(mode) + ' ################################')

    alphas = params[mode]['alphas']
    print(alphas)

    # Calculate final gust
    # Calculate final gust
    gust = zvp10 + alphas['1']*tcm*zvp10 + alphas['2']*tcm*zvp10**2
    maxid = gust.argmax(axis=1)
    I = np.indices(maxid.shape)
    gust_max = gust[I,maxid].squeeze()

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


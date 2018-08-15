import os
import copy
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import pickle
from functions import plot_error
import globals as G
from namelist_cases import Case_Namelist
from functions_train import stat_calculate_gust

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
#####################################

# create directories
if i_plot > 1 and not os.path.exists(CNapply.plot_path):
    os.mkdir(CNapply.plot_path)


# LOAD PARAMS
params = pickle.load( open(CNtrain.params_stat_path, 'rb') )
#print(params)
#quit()


# LAOD DATA
data = pickle.load( open(CNapply.train_stat_path, 'rb') )
model_mean = data['model_mean']
obs_gust = data['obs_gust']
obs_mean = data['obs_mean']
features = data['features']
feature_names = data['feature_names']


# obs nan mask
obsmask = np.isnan(obs_gust)
obsmask[np.isnan(obs_mean)] = True

obs_gust = obs_gust[~obsmask] 
obs_mean = obs_mean[~obsmask]
model_mean = model_mean[~obsmask]
for feat in feature_names:
    features[feat] = features[feat][~obsmask]

# initial gust
gust = features['zvp10'] + 7.2*features['tcm']*features['zvp10']
gust_max_original = np.amax(gust,axis=1)

zvp10 = copy.deepcopy(features['zvp10'])

for mode in params.keys():
    print('#################################################################################')
    print('############################## ' + str(mode) + ' ################################')

    alphas = params[mode]
    print(alphas)

    # Calculate final gust
    gust = stat_calculate_gust(mode, features, alphas, zvp10)
    maxid = gust.argmax(axis=1)
    I = np.indices(maxid.shape)
    gust_max = gust[I,maxid].squeeze()

    plot_error(obs_gust, model_mean, obs_mean, gust_max, gust_max_original)
    plt.suptitle('apply STAT  '+mode)

    if i_plot == 1:
        plt.show()
    elif i_plot > 1:
        if i_label == '':
            plot_name = CNapply.plot_path + 'applied_stat_sw_'+i_sample_weight+'_'+str(mode)+'.png'
        else:
            plot_name = CNappyl.plot_path + 'applied_stat_sw_'+i_sample_weight+'_'+str(i_label)+'_'+str(mode)+'.png'
        print(plot_name)
        plt.savefig(plot_name)
        plt.close('all')


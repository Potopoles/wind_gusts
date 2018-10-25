import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_functions import plot_error, plot_type1
from training_functions import (find_hourly_max, apply_scaling, fill_dict,
                                write_performance_report)
import globals as G
from namelist_cases import Case_Namelist
import namelist_cases as nl
from datetime import timedelta
from netCDF4 import Dataset
from predictors import Predictors
from training import train_linear_model
from linear_models import Linear_Models
import multiprocessing as mp

############ USER INPUT #############
train_case_index = nl.train_case_index
apply_case_index = nl.apply_case_index
CNtrain = Case_Namelist(train_case_index)
CNapply = Case_Namelist(apply_case_index)
# do not plot (0) show plot (1) save plot (2)
i_plot = nl.i_plot
i_plot_type = nl.i_plot_type
reset_model_constellation = nl.apply_reset_model_constellation
#####################################

###############################################################################
###### PART 0: General Setup
###############################################################################
print('########################### PART 0: General Setup')

# create directories
if i_plot > 1 and not os.path.exists(CNapply.plot_path):
    os.mkdir(CNapply.plot_path)

# load data
data = pickle.load( open(CNapply.mod_path, 'rb') )
ncf = Dataset(CNapply.mod_nc_path, 'r')

# load trained linear models
lms_stored = pickle.load( open(CNtrain.output_binary, 'rb') )
#print(lms_stored['models'])
#print(lms_stored['coefs'])

# load Predictor class and linear model (lm)
PR = Predictors(ncf, data)
predictor_list = []
run_which_models = []
keys = []
c = 0
for model_key,model in lms_stored['models'].items():
    print(model_key)
    predictor_list.append(model)
    keys.append(int(model_key))

    run_which_models.append(c)
    c += 1
# order predictor_list according to key numbers
keys, predictor_list = zip(*sorted(zip(keys, predictor_list), reverse=False))

# load linear model
lms = Linear_Models(PR, CNapply, reset_model_constellation, run_which_models,
                    predictor_list)

# data that must be loaded in any case.
obs_mean = data['obs_mean']
obs_gust = data['obs_gust']
zvp10 = np.ma.filled(ncf['zvp10'][:], fill_value=np.nan) 
tcm = PR.preproc['tcm']('tcm')

## observation to 1D and filter values
model_mean = np.mean(zvp10, axis=2)

# Remove values with no data
# get obs nan mask
obsmask = np.isnan(obs_gust)
obsmask[np.isnan(obs_mean)] = True
# remove data in predictors
obs_gust = obs_gust[~obsmask] 
obs_mean = obs_mean[~obsmask]
model_mean = model_mean[~obsmask]
tcm = tcm[~obsmask]
zvp10 = zvp10[~obsmask]

# Calculate reference gust
gust_ref = zvp10 + 7.2*zvp10*tcm
gust_max_ref = find_hourly_max(gust_ref)


###############################################################################
###### PART 1: Load and Calculate and Preprocess Predictors
###############################################################################
print('########################### PART 1: Load Predictors')

predictors = {}
scales = {}
for model_key,model in lms.models.items():
    for pred_name in model.keys():
        #print(predictors.keys())
        if not pred_name in predictors.keys():

            # calculate predictor values
            for flI,field in enumerate(model[pred_name]['prod']):

                # Loading, calculating and preprocessing of fields
                if field[0] in PR.preproc.keys():
                    field_values = PR.preproc[field[0]](field[0])
                else:
                    raise NotImplementedError('Predictor structure: The field '+\
                            field[0] + ' is not defined.')

                # Raise to the power of field[1] and
                # multiply with other fields of current predictors
                if flI == 0:
                    pred_values = field_values**field[1]
                else:
                    pred_values = pred_values * field_values**field[1]

            # remove data where masked due observation missing values
            pred_values = pred_values[~obsmask]

            # transform
            if 'transform' in model[pred_name].keys():
                raise NotImplementedError()
                #pred_values[pred_values < 0.1] = 0.1
                #pred_values = np.log(pred_values)

            # store
            predictors[pred_name] = pred_values


# LOAD COEFFICIENTS 
rm_keys = []
for model_key,lm in lms.models.items():
    try:
        lms.coefs[model_key] = lms_stored['coefs'][model_key]
    # if model does not exist because training failed: remove it
    except KeyError:
        print('WARNING: Model with key ' + model_key + ' does not exist.')
        rm_keys.append(model_key)
for model_key in rm_keys:
    del lms.models[model_key]


###############################################################################
###### PART 2: Plot and Scores
###############################################################################
print('########################### PART 2: Plot and Scores')



for model_key,lm in lms.models.items():
    print(model_key)

    #print(lms.coefs[model_key])
    #print(lms.models[model_key])
    #quit()

    lm_predictors = {}
    for pred_name in lm.keys():
        lm_predictors[pred_name] = np.copy(predictors[pred_name])

    # calculate timestep gusts
    gust = np.zeros(lm_predictors[next(iter(lm_predictors))].shape)
    #print(lm_predictors.keys())
    for pred_name in lm_predictors.keys():
        if lm[pred_name]['fix']:
            gust += lm_predictors[pred_name]
        else:
            gust += lms.coefs[model_key][pred_name] * lm_predictors[pred_name]

    # calculate current hourly gusts
    gust_max = find_hourly_max(gust)
    #quit()



    # PLOT
    if i_plot_type == 0:
        plot_error(obs_gust, model_mean, obs_mean, gust_max, gust_max_ref)
    elif i_plot_type == 1:
        (scores_ref, scores) = plot_type1(obs_gust, gust_max,
                                gust_max_ref, obs_mean, model_mean)
    plt.suptitle('Linear Model '+model_key)

    lms_stored['scores'][model_key] = scores

    if i_plot == 1:
        plt.show()
    elif i_plot > 1:
        if i_plot_type == 0:
            plot_name = CNapply.plot_path + model_key + '.png'
        elif i_plot_type == 1:
            plot_name = CNapply.plot_path + 'plot1_' + model_key + '.png'
        print(plot_name)
        plt.savefig(plot_name)
        plt.close('all')
    #quit()


###############################################################################
###### PART 3: Output
###############################################################################
print('########################### PART 3: Output')

# Update Performance Report
score_names = ['me', 'rmse', 'corr', 'pod20', 'pod40', 'far20', 'far40']
reverses =    [False, False, True,   True,     True,     False,  False ]
for i in range(0,len(score_names)):
    write_performance_report(CNapply, lms_stored, score_names[i], reverses[i],
                            scores_ref)


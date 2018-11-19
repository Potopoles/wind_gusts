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
case_index = nl.case_index
CN = Case_Namelist(case_index)
# do not plot (0) show plot (1) save plot (2)
i_plot = nl.i_plot
i_plot_type = nl.i_plot_type
max_tuning_steps = 2000
coef_conv_thresh = 1E-3
weights_err_spaces = {'1_1':0,'err':1}
nth_ts_out = 20
reset_model_constellation = nl.reset_model_constellation
i_transform = 0
if i_transform:
    n_bins = 2
    weight_slope = 0/n_bins
else:
    n_bins = 4
    weight_slope = 4/n_bins

if len(sys.argv) > 1:
    run_which_models = [int(sys.argv[1])]
    print('Run model number ' + str(sys.argv[1]) + '.')
else:
    run_which_models = [0]
    print('Model number not given. Default is ' + str(0) + '.')
#####################################

###############################################################################
###### PART 0: General Setup
###############################################################################
print('########################### PART 0: General Setup')

# create directories
if i_plot > 1 and not os.path.exists(CN.plot_path):
    os.mkdir(CN.plot_path)

# load data
data = pickle.load( open(CN.mod_path, 'rb') )
ncf = Dataset(CN.mod_nc_path, 'r')

# load Predictor class and linear model (lm)
PR = Predictors(ncf, data)
lms = Linear_Models(PR, CN, reset_model_constellation, run_which_models)

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

#obs_gust[obs_gust < 0.1] = 0.1
#obs_gust = np.log(obs_gust)
#obs_gust = np.sqrt(obs_gust)

# Calculate reference gust
gust_ref = zvp10 + 7.2*zvp10*tcm
gust_max_ref = find_hourly_max(gust_ref)


if i_transform:
    obs_gust[obs_gust == 0] = 0.1
    obs_gust = np.log(obs_gust)


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

            if i_transform:
                if np.min(pred_values) < 0:
                    raise ValueError('Smallest values is below 0')
                pred_values[pred_values == 0] = np.min(pred_values)
                pred_values = np.log(pred_values)

            # apply scaling
            if model[pred_name]['fix'] is not 1:
                pred_values,sd = apply_scaling(pred_values)
                scales[pred_name] = sd
            else:
                scales[pred_name] = 1

            # store
            predictors[pred_name] = pred_values



###############################################################################
###### PART 2: Training and Output
###############################################################################
print('########################### PART 2: Train')

for model_key,lm in lms.models.items():

    lm_predictors = {}
    for pred_name in lm.keys():
        lm_predictors[pred_name] = np.copy(predictors[pred_name])

    result = train_linear_model(model_key, lm, lm_predictors,
                        obs_gust, obs_mean,
                        gust_max_ref, model_mean, i_transform,
                        n_bins, weight_slope, max_tuning_steps,
                        weights_err_spaces, coef_conv_thresh, nth_ts_out,
                        i_plot, i_plot_type, plot_type1, CN)
    lms.coefs[model_key]    = result[0]
    scores_ref              = result[1]
    lms.scores[model_key]   = result[2]

print('Finished Tuning!')
###############################################################################
###### PART 3: Output
###############################################################################
print('########################### PART 3: Output')

# Unscaling
for model_key,lm in lms.models.items():
    for pred_name in lm.keys():
        lms.coefs[model_key][pred_name] /= scales[pred_name]
    print('Unscaled coefficients: ')
    print(lms.coefs[model_key])


# SAVE MODEL PARAMETERS 
if not os.path.exists(CN.output_path):
    os.mkdir(CN.output_path)
if not os.path.exists(CN.output_binary):
    data_out = {}
    data_out['models'] = {}
    data_out['coefs'] = {}
    data_out['scores'] = {}
    pickle.dump(data_out, open(CN.output_binary, 'wb'))
data_out = pickle.load( open(CN.output_binary, 'rb') )
for model_key,lm in lms.models.items():
    data_out['models'][model_key] = list(lm.keys())
    data_out['coefs'][model_key] = lms.coefs[model_key]
    data_out['scores'][model_key] = lms.scores[model_key]
pickle.dump(data_out, open(CN.output_binary, 'wb'))


# Update Coefficient Report
file_name = CN.output_path + 'coefs.txt'
with open(file_name, 'w') as f:
    #for model_key,lm in lms.models.items():
    for model_key in data_out['coefs'].keys():
        f.write('{}\t {}\n'.format(model_key, data_out['coefs'][model_key]))

# Update Performance Report
score_names = ['me', 'rmse', 'corr', 'pod20', 'pod40', 'far20', 'far40']
reverses =    [False, False, True,   True,     True,     False,  False ]
for i in range(0,len(score_names)):
    write_performance_report(CN, data_out, score_names[i], reverses[i],
                            scores_ref)


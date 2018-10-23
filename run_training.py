import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from plot_functions import plot_error, plot_type1
from training_functions import (find_hourly_max, apply_scaling, fill_dict)
import globals as G
from namelist_cases import Case_Namelist
import namelist_cases as nl
from datetime import timedelta
from netCDF4 import Dataset
from predictors import Predictors
from training import train_linear_model
from linear_models import Linear_Models

############ USER INPUT #############
case_index = nl.case_index
CN = Case_Namelist(case_index)
# do not plot (0) show plot (1) save plot (2)
i_plot = nl.i_plot
i_plot_type = nl.i_plot_type
model_dt = nl.model_dt
nhrs_forecast = nl.nhrs_forecast
#i_output_error = 1
max_tuning_steps = 1000
coef_conv_thresh = 1E-2
n_bins = 2
weight_slope = 4/n_bins
weights_err_spaces = {'1_1':0.5,'err':1}
nth_ts_out = 50


#####################################

###############################################################################
###### PART 0: General Setup
###############################################################################

# create directories
if i_plot > 1 and not os.path.exists(CN.plot_path):
    os.mkdir(CN.plot_path)


# load data
data = pickle.load( open(CN.mod_path, 'rb') )
ncf = Dataset(CN.mod_nc_path, 'r')

# load Predictor class and linear model (lm)
lms = Linear_Models()
PR = Predictors(ncf)

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

predictors = {}
scales = {}
for model_key,model in lms.models.items():
    for pred_name in model.keys():
        if not pred_name in predictors.keys():

            # calculate predictor values
            for flI,field in enumerate(model[pred_name]['prod']):

                # Loading, calculating and preprocessing of fields
                if field[0] in PR.preproc.keys():
                    field_values = PR.preproc[field[0]](field[0])
                else:
                    raise NotImplementedError()

                # Raise to the power of field[1] and
                # multiply with other fields of current predictors
                if flI == 0:
                    pred_values = field_values**field[1]
                else:
                    pred_values = pred_values * field_values**field[1]

            # remove data where masked due observation missing values
            pred_values = pred_values[~obsmask]

            # apply scaling
            if model[pred_name]['fix'] is not 1:
                pred_values,sd = apply_scaling(pred_values)
                scales[pred_name] = sd
            else:
                scales[pred_name] = 1

            # store
            predictors[pred_name] = pred_values

    #print(predictors.keys())


###############################################################################
###### PART 2: Training and Output
###############################################################################

for model_key,lm in lms.models.items():
    
    #lm_local = lms.models[model_key]
    lm_predictors = {}
    for pred_name in lm.keys():
        lm_predictors[pred_name] = np.copy(predictors[pred_name])
    print(lm.keys())

    result = train_linear_model(lm, lm_predictors, obs_gust, obs_mean,
                        gust_max_ref, model_mean,
                        n_bins, weight_slope, max_tuning_steps,
                        weights_err_spaces, coef_conv_thresh, nth_ts_out,
                        i_plot, i_plot_type, plot_type1)
    lms.coefs[model_key] = result[0]

for model_key,lm in lms.models.items():
    print(lms.coefs[model_key])
quit()

###############################################################################
###### PART 3: Output
###############################################################################

# Unscaling
for pred_name in predictors.keys():
    predictors[pred_name] = predictors[pred_name] * scales[pred_name]
    coefs[pred_name] = coefs[pred_name] / scales[pred_name]
print('Final Unscaled Coefficients')
print(str(coefs))




quit()

# SAVE PARAMETERS 
params = {}
params[mode] = {'alphas':{'1':alpha1,'2':alpha2}}
pickle.dump(params, open(CN.params_readj_path, 'wb'))



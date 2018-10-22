import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from plot_functions import plot_error, plot_type1
from tuning_functions import (find_hourly_max, calc_bins, apply_scaling)
import globals as G
from namelist_cases import Case_Namelist
import namelist_cases as nl
from datetime import timedelta
from netCDF4 import Dataset

############ USER INPUT #############
case_index = nl.case_index
CN = Case_Namelist(case_index)
# do not plot (0) show plot (1) save plot (2)
i_plot = nl.i_plot
i_plot_type = nl.i_plot_type
model_dt = nl.model_dt
nhrs_forecast = nl.nhrs_forecast
i_output_error = 1

modes = ['ln',
         'nl']

i_mode_ints = range(0,len(modes))
i_mode_ints = [0]
# TODO debug
mode = 'test'
#####################################

###############################################################################
###### PART 0: General Setup
###############################################################################

lm = {}
#lm['zvp10']                 = {'fix':1,
#                                   'prod':[('zvp10',1)]
#                                  }
lm['zvp10']                 = {'fix':1,
                                   'prod':[('zvp10',1)]
                                  }
lm['zvp10_tcm']             = {'fix':0,
                                   'prod':[('zvp10',1),('tcm',1)]
                                  }
lm['bra']                  = {'fix':0,
                                   'prod':[('zv_bra_es',1)]
                                  }
#lm['bra']                  = {'fix':0,
#                                   'prod':[('zv_bra_es',1),('zvp10',1)]
#                                  }


# create directories
if i_plot > 1 and not os.path.exists(CN.plot_path):
    os.mkdir(CN.plot_path)

def fill_dict(predictors, value=0):
    dict = {}
    for pred_name in predictors.keys():
        dict[pred_name] = value
    return(dict)

###############################################################################
###### PART 1: Read in Fields and Preprocessing
###############################################################################

# load data
data = pickle.load( open(CN.mod_path, 'rb') )
ncf = Dataset(CN.mod_nc_path, 'r')

# data that must be loaded in any case.
obs_mean = data['obs_mean']
obs_gust = data['obs_gust']
zvp10 = np.ma.filled(ncf['zvp10'][:], fill_value=np.nan) 
tcm = np.ma.filled(ncf['tcm'][:], fill_value=np.nan) 
tcm[tcm < 5E-4] = 5E-4
tcm = np.sqrt(tcm)

predictors = {}
for pred_name in lm.keys():
    for flI,field in enumerate(lm[pred_name]['prod']):
        field_values = np.ma.filled(ncf[field[0]][:], fill_value=np.nan)

        # Preprocessing of tcm
        if field[0] == 'tcm':
            field_values[field_values < 5E-4] = 5E-4
            field_values = np.sqrt(field_values)

        if flI == 0:
            pred_values = field_values**field[1]
        else:
            pred_values = pred_values * field_values**field[1]
    predictors[pred_name] = pred_values


## observation to 1D and filter values
model_mean = np.mean(zvp10, axis=2)

# Remove values with no data
# get obs nan mask
obsmask = np.isnan(obs_gust)
obsmask[np.isnan(obs_mean)] = True
# remove data in all predictors
obs_gust = obs_gust[~obsmask] 
obs_mean = obs_mean[~obsmask]
model_mean = model_mean[~obsmask]
tcm = tcm[~obsmask]
zvp10 = zvp10[~obsmask]
for pred_name in predictors.keys():
    predictors[pred_name] = predictors[pred_name][~obsmask]

# Calculate reference gust
gust_ref = zvp10 + 7.2*zvp10*tcm
gust_max_ref = find_hourly_max(gust_ref)


# Scaling
scales = fill_dict(predictors)
for pred_name in predictors.keys():
    predictors[pred_name],sd = apply_scaling(predictors[pred_name])
    scales[pred_name] = sd

###############################################################################
###### PART 2: Training
###############################################################################

coefs = fill_dict(predictors, 0)
learning_rate = fill_dict(predictors, 1E-2)
n_bins = 4
weight_slope = 2
weights_err_spaces = {'1_1':0.3,'err':1}

bins, bin_weights = calc_bins(n_bins, weight_slope=weight_slope)

sin_th = np.sin(-np.pi/4)
cos_th = np.cos(-np.pi/4)

N = len(obs_gust)

for learning_step in range(0,200):
    
    # calculate current timestep gusts
    #gust = predictors['zvp10'] + alpha*predictors['zvp10_tcm']
    gust = np.zeros(predictors[next(iter(predictors))].shape)
    for pred_name in predictors.keys():
        if lm[pred_name]['fix']:
            gust += predictors[pred_name]
        else:
            gust += coefs[pred_name] * predictors[pred_name]

    # calculate current hourly gusts
    gust_max, predictors_max = find_hourly_max(gust, predictors)

    # calculate (error) fields for gradient descent
    speed_1_1_space   =    - sin_th * gust_max + cos_th * obs_gust
    dev_1_1_space = - (  sin_th * gust_max + cos_th * obs_gust )
    dev_err_space = - (  gust_max - obs_gust                   )

    ##########################################################################
    ##### LOOP OVER BINS AND CALCULATE GRADIENT OF COST FUNCTION
    ##########################################################################
    tot_weight_1_1_space = fill_dict(predictors, 0)
    tot_weight_err_space = fill_dict(predictors, 0)
    dalpha_1_1_space     = fill_dict(predictors, 0)
    dalpha_err_space     = fill_dict(predictors, 0)
    for bI in range(0,n_bins):
        #bI = 4

        ##### GRADIENT DESCENT IN ERROR SPACE
        ######################################################################
        if weights_err_spaces['err'] > 0:
            bin_inds = np.argwhere((obs_gust >= bins['err'][bI][0]) & \
                                   (obs_gust < bins['err'][bI][1])).squeeze()
        else:
            bin_inds = []
        mod_err_bin = dev_err_space[bin_inds]
        try:
            N_bin = len(mod_err_bin)
        except TypeError: # in case model_error_bin is scalar not array
            N_bin = 1
        if N_bin > 20:
            predictors_bin = {}
            #predictors_bin['zvp10_tcm'] = predictors_max['zvp10_tcm'][bin_inds]
            for pred_name in predictors.keys():
                if not lm[pred_name]['fix']:
                    predictors_bin[pred_name] = \
                                    predictors_max[pred_name][bin_inds]
                    # calculate derivative of loss function with respect
                    # to alpha
                    dalpha_err_space[pred_name] += bin_weights[bI] * \
                                  2/N_bin * np.sum(mod_err_bin *
                                                predictors_bin[pred_name])
                    tot_weight_err_space[pred_name] += bin_weights[bI]


        ##### GRADIENT DESCENT IN 1_1 SPACE
        ######################################################################
        if weights_err_spaces['1_1'] > 0:
            bin_inds = np.argwhere((speed_1_1_space >= bins['1_1'][bI][0]) & \
                                   (speed_1_1_space < bins['1_1'][bI][1])).\
                                                                    squeeze()
        else:
            bin_inds = []
        mod_err_bin = dev_1_1_space[bin_inds]
        try:
            N_bin = len(mod_err_bin)
        except TypeError: # in case model_error_bin is scalar not array
            N_bin = 1
        if (N_bin > 20) and weights_err_spaces['1_1'] > 0:
            predictors_bin = {}
            #predictors_bin['zvp10_tcm'] = predictors_max['zvp10_tcm'][bin_inds]
            for pred_name in predictors.keys():
                if not lm[pred_name]['fix']:
                    predictors_bin[pred_name] = \
                                    predictors_max[pred_name][bin_inds]
                    # calculate derivative of loss function with respect
                    # to alpha
                    dalpha_1_1_space[pred_name] += bin_weights[bI] * \
                                  2/N_bin * np.sum(mod_err_bin * sin_th *
                                                predictors_bin[pred_name])
                    tot_weight_1_1_space[pred_name] += bin_weights[bI]


    ##########################################################################
    ##### UPDATE COEFFICIENTS
    ##########################################################################
    for pred_name in predictors.keys():
        if not lm[pred_name]['fix']:
            try:
                dalpha_1_1_space[pred_name] = (dalpha_1_1_space[pred_name] / 
                                            tot_weight_1_1_space[pred_name])
            except ZeroDivisionError:
                dalpha_1_1_space[pred_name] = 0.
            try:
                dalpha_err_space[pred_name] = (dalpha_err_space[pred_name] /
                                            tot_weight_err_space[pred_name])
            except ZeroDivisionError:
                dalpha_err_space[pred_name] = 0.

            # calculate weighted combination of dalpha from 1_1 space and err
            # space
            dcoef = (weights_err_spaces['err']*dalpha_err_space[pred_name] + 
                     weights_err_spaces['1_1']*dalpha_1_1_space[pred_name] ) / \
                    (weights_err_spaces['err'] + weights_err_spaces['1_1'])

            coefs[pred_name] += learning_rate[pred_name] * dcoef

    if learning_step % 5 == 0:
        print('coefficients ' + str(coefs))




###############################################################################
###### PART 3: Output
###############################################################################

# calculate final timestep gusts
gust = np.zeros(predictors[next(iter(predictors))].shape)
for pred_name in predictors.keys():
    if lm[pred_name]['fix']:
        gust += predictors[pred_name]
    else:
        gust += coefs[pred_name] * predictors[pred_name]

# calculate current hourly gusts
gust_max = find_hourly_max(gust)


# PLOT
if i_plot > 0:
    if i_plot_type == 0:
        plot_error(obs_gust, model_mean, obs_mean, gust_max, gust_max_ref)
    elif i_plot_type == 1:
        plot_type1(obs_gust, gust_max, gust_max_ref, obs_mean, model_mean)
    else:
        raise NotImplementedError()
    plt.suptitle('READJUST  '+mode)

    if i_plot == 1:
        plt.show()
    elif i_plot > 1:
        if i_plot_type == 0:
            plot_name = CN.plot_path + 'tuning_readj_'+str(mode)+'.png'
        elif i_plot_type == 1:
            plot_name = CN.plot_path + 'plot1_tuning_readj_'+str(mode)+'.png'
        print(plot_name)
        plt.savefig(plot_name)
        plt.close('all')


quit()
# RESCALE ALPHA VALUES
# not scaled

# SAVE PARAMETERS 
params = {}
params[mode] = {'alphas':{'1':alpha1,'2':alpha2}}
pickle.dump(params, open(CN.params_readj_path, 'wb'))



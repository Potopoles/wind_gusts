from training_functions import (find_hourly_max, calc_bins, apply_scaling,
                                fill_dict)
import numpy as np
import matplotlib.pyplot as plt
from predictors import Predictors


def train_linear_model(model_key, lm, predictors,
                obs_gust, obs_mean,
                gust_max_ref, model_mean,
                n_bins, weight_slope, max_tuning_steps,
                weights_err_spaces, coef_conv_thresh, nth_ts_out,
                i_plot, i_plot_type, plot_type1, CN):

    ###########################################################################
    ###### PART 2: Training
    ###########################################################################

    coefs = fill_dict(lm, 0)
    for pred_name in lm.keys():
        if lm[pred_name]['fix']:
            coefs[pred_name] = 1
    learning_rate = fill_dict(lm, 3E-3)

    bins, bin_weights = calc_bins(n_bins, weight_slope=weight_slope)

    sin_th = np.sin(-np.pi/4)
    cos_th = np.cos(-np.pi/4)

    N = len(obs_gust)

    dcoefs = fill_dict(lm, np.full(3, np.nan))

    for learning_step in range(0,max_tuning_steps):
        
        # calculate current timestep gusts
        gust = np.zeros(predictors[next(iter(predictors))].shape)
        for pred_name in lm.keys():
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

        #######################################################################
        ##### LOOP OVER BINS AND CALCULATE GRADIENT OF COST FUNCTION
        #######################################################################
        tot_weight_1_1_space = fill_dict(lm, 0)
        tot_weight_err_space = fill_dict(lm, 0)
        dalpha_1_1_space     = fill_dict(lm, 0)
        dalpha_err_space     = fill_dict(lm, 0)
        for bI in range(0,n_bins):
            #bI = 4

            ##### GRADIENT DESCENT IN ERROR SPACE
            ###################################################################
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
                for pred_name in lm.keys():
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
            ###################################################################
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
                for pred_name in lm.keys():
                    if not lm[pred_name]['fix']:
                        predictors_bin[pred_name] = \
                                        predictors_max[pred_name][bin_inds]
                        # calculate derivative of loss function with respect
                        # to alpha
                        dalpha_1_1_space[pred_name] += bin_weights[bI] * \
                                      2/N_bin * np.sum(mod_err_bin * sin_th *
                                                    predictors_bin[pred_name])
                        tot_weight_1_1_space[pred_name] += bin_weights[bI]


        #######################################################################
        ##### UPDATE COEFFICIENTS
        #######################################################################
        training_complete = True
        for pred_name in lm.keys():
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

                # Check if gradient descent runs without oscillations
                dcoefs[pred_name] = np.roll(dcoefs[pred_name], shift=1)
                dcoefs[pred_name][0] = dcoef*learning_rate[pred_name]
                deriv2_coef_change = dcoefs[pred_name][0] + \
                            dcoefs[pred_name][2] - 2*dcoefs[pred_name][1]
                abs_coef_change = np.nanmean(np.abs(dcoefs[pred_name]))
                if np.abs(deriv2_coef_change) > abs_coef_change:
                    learning_rate[pred_name] *= 0.95
                    #print('Attention: Oszillations!')
                    #print(pred_name + ' lr ' + str(learning_rate[pred_name]))
                elif np.abs(deriv2_coef_change)*10 < abs_coef_change:
                    learning_rate[pred_name] *= 1.02
                    #print(pred_name + ' lr ' + str(learning_rate[pred_name]))
                #if learning_step % 5 == 0:
                #    print(str(deriv2_coef_change) + '\t' + str(abs_coef_change))
                #print(abs_coef_change)

                # check if coefficient converged already
                if abs_coef_change > coef_conv_thresh:
                    training_complete = False

                coefs[pred_name] += learning_rate[pred_name] * dcoef

        if learning_step % nth_ts_out == 0:
            print(model_key + '\t' + str(learning_step) + '\t ' + str(coefs))

        # if all coefficients already converged, finish training.
        if training_complete:
            print('Coefficients converged. Took ' + str(learning_step) + ' steps')
            break


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

    #gust_max = np.exp(gust_max)
    #obs_gust = np.exp(obs_gust)
    #gust_max = gust_max**2
    #obs_gust = obs_gust**2

    # PLOT
    if i_plot_type == 0:
        plot_error(obs_gust, model_mean, obs_mean, gust_max, gust_max_ref)
    elif i_plot_type == 1:
        (errors_ref, errors) = plot_type1(obs_gust, gust_max,
                                gust_max_ref, obs_mean, model_mean)
    plt.suptitle('Linear Model '+model_key)

    if i_plot == 1:
        plt.show()
    elif i_plot > 1:
        if i_plot_type == 0:
            plot_name = CN.plot_path + model_key + '.png'
        elif i_plot_type == 1:
            plot_name = CN.plot_path + 'plot1_' + model_key + '.png'
        print(plot_name)
        plt.savefig(plot_name)
        plt.close('all')



    result = (coefs, errors_ref, errors)

    return(result)

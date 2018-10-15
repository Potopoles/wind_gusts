import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from functions import plot_error, plot_mod_vs_obs
import globals as G
from namelist_cases import Case_Namelist
import namelist_cases as nl
from datetime import timedelta

############ USER INPUT #############
case_index = nl.case_index
CN = Case_Namelist(case_index)
# do not plot (0) show plot (1) save plot (2)
i_plot = nl.i_plot
i_plot_type = nl.i_plot_type
model_dt = nl.model_dt
nhrs_forecast = nl.nhrs_forecast
i_load = nl.i_load
i_train = nl.i_train
i_output_error = 1
default_learning_rate_factor = 1E-2
delete_existing_param_file = nl.delete_existing_param_file
#max_mean_wind_error = nl.max_mean_wind_error
#sample_weight = nl.sample_weight

modes = ['ln',
         'nl']

i_mode_ints = range(0,len(modes))
i_mode_ints = [0]
sgd_prob = 1.0

# binning
gust_bins = [(0,10),(10,20),(20,30),(30,40),(40,50)]
bin_weights = [0,0,0,0,1]
#gust_bins = [(0,20),(20,40),(40,60)]
#bin_weights = [1,2,10]
gust_bins = [(0,10),(10,20),(20,30),(30,40),(40,50),(50,60)]
bin_weights = [1,0,0,0,0,0]
n_bins = len(gust_bins)
n_iter = 1
#####################################

# create directories
if i_plot > 1 and not os.path.exists(CN.plot_path):
    os.mkdir(CN.plot_path)

if delete_existing_param_file:
    try:
        os.remove(CN.params_readj_path)
    except:
        pass


if not i_load:
    # load data
    data = pickle.load( open(CN.mod_path, 'rb') )

    stat_keys = data[G.STAT_NAMES]

    lm_runs = list(data[G.MODEL][G.STAT][stat_keys[0]][G.RAW].keys())
    n_hours = len(lm_runs)*nhrs_forecast
    n_stats = len(stat_keys)
    ts_per_hour = int(3600/model_dt)


    # 3D
    tcm = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    zvp10 = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    print('3D shape ' + str(tcm.shape))
    # 2D
    obs_gust = np.full((n_hours, n_stats), np.nan)
    obs_mean = np.full((n_hours, n_stats), np.nan)

    # hour indices within one lm_run
    hour_inds = np.arange(0,nhrs_forecast).astype(np.int)
    for lmi,lm_run in enumerate(lm_runs):
        print('\n' + lm_run)
        lm_inds = np.arange(lmi*nhrs_forecast,(lmi+1)*nhrs_forecast)

        for si,stat_key in enumerate(stat_keys):
            if si % (int(len(stat_keys)/10)+1) == 0:
                print(str(int(100*si/len(stat_keys))), end='\t', flush=True)

            for hi in hour_inds:
                hr_ind = lm_inds[hi]

                ts_inds = np.arange(hi*ts_per_hour,(hi+1)*ts_per_hour).astype(np.int)

                tcm[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['tcm'][ts_inds]
                zvp10[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['zvp10'][ts_inds]

            # OBSERVATION DATA
            full_hr_timestamps = data[G.MODEL][G.STAT][stat_keys[0]][G.RAW][lm_run]\
                                        ['tcm'].resample('H').max().index[1:]
            obs_gust[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_GUST_SPEED][full_hr_timestamps] 
            obs_mean[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_MEAN_WIND][full_hr_timestamps] 


    # Process fields
    tcm[tcm < 5E-4] = 5E-4
    tcm = np.sqrt(tcm)

        
    # observation to 1D and filter values
    model_mean = np.mean(zvp10, axis=2)


    data = {}
    data['model_mean'] = model_mean
    data['obs_gust'] = obs_gust
    data['obs_mean'] = obs_mean 
    data['tcm'] = tcm 
    data['zvp10'] = zvp10 
    data['stations'] = stat_keys 

    pickle.dump(data, open(CN.train_readj_path, 'wb'))

else:

    data = pickle.load( open(CN.train_readj_path, 'rb') )

    model_mean = data['model_mean']
    obs_gust = data['obs_gust']
    obs_mean = data['obs_mean']
    tcm = data['tcm']
    zvp10 = data['zvp10']




#########################
## TEST CASE TO COMPARE WITH test_gusts.py script
#si = 0 # 0: ABO
#gust = zvp10[:,si,:] + 7.2 * tcm[:,si,:] * zvp10[:,si,:]
#gust = np.max(gust,axis=1)
#print(np.round(gust,6))
#quit()
#########################



if i_train:

    # obs nan mask
    obsmask = np.isnan(obs_gust)
    obsmask[np.isnan(obs_mean)] = True
    ## bad mean wind accuracy mask
    #mean_abs_error = np.abs(model_mean - obs_mean)
    #mean_rel_error = mean_abs_error/obs_mean
    #errormask = mean_rel_error > max_mean_wind_error
    ## combine both
    #obsmask[errormask] = True

    obs_gust = obs_gust[~obsmask] 
    obs_mean = obs_mean[~obsmask]
    model_mean = model_mean[~obsmask]
    tcm = tcm[~obsmask]
    zvp10 = zvp10[~obsmask]

    



    for mode_int in i_mode_ints:
        mode = modes[mode_int]
        print('#################################################################################')
        print('############################## ' + str(mode) + ' ################################')

        # initial gust
        if mode == 'ln':
            gust = zvp10 + 7.2*tcm*zvp10
        elif mode == 'nl':
            gust = zvp10 + 7.2*tcm*zvp10 + 0.09*tcm*zvp10**2
        gust_max_orig = np.amax(gust,axis=1)

        if mode == 'nl':
            learning_rate_factor = default_learning_rate_factor * 1/20
            d_error_thresh = 1E-6
        else:
            learning_rate_factor = default_learning_rate_factor
            d_error_thresh = 1E-5


        alpha1 = 7
        alpha2 = 0
        error_old = np.Inf
        d_errors = np.full(int(1/sgd_prob*5), 100.)
        #learning_rate = 1E-5
        learning_rate = 1E-1

        c = 0
        #while np.abs(np.mean(d_errors)) > d_error_thresh:
        while c < n_iter:
            print('############')


            rms_bins = []

            # SGD selection
            sgd_inds = np.random.choice([True, False], (zvp10.shape[0]), p=[sgd_prob,1-sgd_prob])
            sgd_zvp10 = zvp10[sgd_inds,:]
            sgd_tcm = tcm[sgd_inds,:]
            sgd_obs_gust = obs_gust[sgd_inds]

            # calc current time step gusts
            if mode == 'ln':
                sgd_gust = sgd_zvp10 + alpha1*sgd_tcm*sgd_zvp10
            elif mode == 'nl':
                sgd_gust = sgd_zvp10 + alpha1*sgd_tcm*sgd_zvp10 + alpha2*sgd_tcm*sgd_zvp10**2
            else:
                raise ValueError('wrong mode')

            # find maximum gust
            maxid = sgd_gust.argmax(axis=1)
            I = np.indices(maxid.shape)
            sgd_tcm_max = sgd_tcm[I,maxid]
            sgd_zvp10_max = sgd_zvp10[I,maxid]
            sgd_gust_max = sgd_gust[I,maxid]
            

            dalpha1 = 0
            for bI in range(0,n_bins):
                #print('bin number: \t' + str(bI))
                bin_inds = np.argwhere((sgd_obs_gust >= gust_bins[bI][0]) & (sgd_obs_gust < gust_bins[bI][1])).squeeze()
                bin_gust_max = sgd_gust_max.squeeze()[bin_inds]
                bin_obs_gust = sgd_obs_gust.squeeze()[bin_inds]
                bin_tcm_max = sgd_tcm_max.squeeze()[bin_inds]
                bin_zvp10_max = sgd_zvp10_max.squeeze()[bin_inds]
                N = len(bin_obs_gust)
                #print('n samples: \t' + str(N))

                # error
                deviation = bin_obs_gust - bin_gust_max
                error_now = np.sqrt(np.sum(deviation**2)/N)
                rms_bins.append(error_now)
                d_error = error_old - error_now
                d_errors = np.roll(d_errors, shift=1)
                d_errors[0] = d_error
                error_old = error_now
                #if i_output_error:
                #    if c % 10 == 0:
                #        print(str(c) + '\t' + str(error_now) + '\t' + str(np.abs(np.mean(d_errors))))
                #        print('alpha 1 ' + str(alpha1) + ' alpha 2 ' + str(alpha2))

                # gradient of parameters
                dalpha1 += bin_weights[bI] * (-2/N * np.sum( bin_tcm_max*bin_zvp10_max * deviation ))
                # poor attempt of absolute loss
                #dalpha1 += bin_weights[bI] * (-1/N * np.sum( bin_tcm_max*bin_zvp10_max ))
                #print(dalpha1)

                #dalpha1 = dalpha1/n_bins
                dalpha1 = dalpha1/np.sum(bin_weights)

            alpha1 = alpha1 - learning_rate * dalpha1
            print('alpha1 ' + str(alpha1))

            # adjust learning rate
            #learning_rate = error_now*learning_rate_factor

            c += 1
            #print('############')

        print('############')
        print('alpha1 ' + str(alpha1))
        print('############')

        print(rms_bins)

    
        alpha1 = 5


        # Calculate final gust
        gust = zvp10 + alpha1*tcm*zvp10

        # find maximum gust
        maxid = gust.argmax(axis=1)
        I = np.indices(maxid.shape)
        gust_max = gust[I,maxid].squeeze()


        # PLOT
        if i_plot > 0:
            if i_plot_type == 0:
                plot_error(obs_gust, model_mean, obs_mean, gust_max, gust_max_orig)
            elif i_plot_type == 1:
                plot_mod_vs_obs(obs_gust, gust_max, gust_max_orig)
            else:
                raise NotImplementedError()
            plt.suptitle('READJUST  '+mode)

            if i_plot == 1:
                plt.show()
            elif i_plot > 1:
                if i_plot_type == 0:
                    plot_name = CN.plot_path + 'bintrain_readj_'+str(mode)+'.png'
                elif i_plot_type == 1:
                    plot_name = CN.plot_path + 'plot1_bintrain_readj_'+str(mode)+'.png'
                print(plot_name)
                plt.savefig(plot_name)
                plt.close('all')


        # RESCALE ALPHA VALUES
        # not scaled

        # SAVE PARAMETERS 
        if os.path.exists(CN.params_readj_path):
            params = pickle.load( open(CN.params_readj_path, 'rb') )
        else:
            params = {}
        params[mode] = {'alphas':{'1':alpha1,'2':alpha2}}
        pickle.dump(params, open(CN.params_readj_path, 'wb'))


else:
    print('Train is turned off. Finish.')

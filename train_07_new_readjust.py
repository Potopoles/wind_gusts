import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from functions import plot_error
import globals as G
from namelist_cases import Case_Namelist
import namelist_cases as nl
from datetime import timedelta

from sklearn.linear_model import LinearRegression

############ USER INPUT #############
case_index = nl.case_index
CN = Case_Namelist(case_index)
# do not plot (0) show plot (1) save plot (2)
i_plot = nl.i_plot
model_dt = nl.model_dt
nhrs_forecast = nl.nhrs_forecast
i_label = ''
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
sgd_prob = 0.1
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
    print()


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


        maxid = (zvp10+7.2*tcm*zvp10).argmax(axis=1)
        I = np.indices(maxid.shape)
        tcm_max = tcm[I,maxid]
        zvp10_max = zvp10[I,maxid]


        X = np.zeros( (tcm_max.shape[1],1) )
        #X[:,1] = zvp10_max[0,:]
        X[:,0] = zvp10_max[0,:] * tcm_max[0,:]
        y = obs_gust - zvp10_max[0,:]

        weights = obs_gust**6

        regr = LinearRegression(fit_intercept=False)
        regr.fit(X,y, sample_weight=weights)

        alpha1 = regr.coef_
        #alpha2 = regr.coef_[1]
        #alpha1 = 10
        alpha2 = 0
        print(alpha1)
        print(alpha2)
        #quit()



        print('############')
        print('alpha1 ' + str(alpha1))
        print('alpha2 ' + str(alpha2))
        print('############')



        # Calculate final gust
        if mode == 'ln':
            gust = zvp10 + alpha1*tcm*zvp10
        elif mode == 'nl':
            gust = zvp10 + alpha1*tcm*zvp10 + alpha2*tcm*zvp10**2
        else:
            raise ValueError('wrong mode')
        # find maximum gust
        maxid = gust.argmax(axis=1)
        I = np.indices(maxid.shape)
        gust_max = gust[I,maxid].squeeze()


        # PLOT
        if i_plot > 0:
            try:
                plot_error(obs_gust, model_mean, obs_mean, gust_max, gust_max_orig)
                plt.suptitle('READJUST  '+mode)

                if i_plot == 1:
                    plt.show()
                elif i_plot > 1:
                    if i_label == '':
                        plot_name = CN.plot_path + 'tuning_readj_'+str(mode)+'.png'
                    else:
                        plot_name = CN.plot_path + 'tuning_readj_'+str(i_label)+'_'+str(mode)+'.png'
                    print(plot_name)
                    plt.savefig(plot_name)
                    plt.close('all')
            except:
                print('Tkinter ERROR while plotting!')


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

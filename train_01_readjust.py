import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from functions import plot_error
import globals as G
from namelist_cases import Case_Namelist
from datetime import timedelta

############ USER INPUT #############
case_index = 10
CN = Case_Namelist(case_index)
# do not plot (0) show plot (1) save plot (2)
i_plot = 2
model_dt = 10
i_label = ''
i_load = 0
i_output_error = 1
default_learning_rate_factor = 1E-2

modes = ['ln',
         'nl']
i_mode_ints = range(0,len(modes))
#i_mode_ints = [1]
max_mean_wind_error = 0.1
max_mean_wind_error = 1.0
max_mean_wind_error = 5.0
max_mean_wind_error = 100.0
delete_existing_param_file = 1
sgd_prob = 0.1

model_time_shift = 1
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
    n_hours = len(lm_runs)*24
    n_stats = len(stat_keys)
    ts_per_hour = int(3600/model_dt)


    # 3D
    tcm = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    zvp10 = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    print('3D shape ' + str(tcm.shape))
    # 2D
    obs_gust = np.full((n_hours, n_stats), np.nan)
    obs_mean = np.full((n_hours, n_stats), np.nan)

    for lmi,lm_run in enumerate(lm_runs):
        print(lm_run)
        lm_inds = np.arange(lmi*24,(lmi+1)*24)
        model_hours_tmp = data[G.MODEL][G.STAT][stat_keys[0]][G.RAW][lm_run]\
                                    ['tcm'].resample('H').max().index
        model_hours_shifted = [hr+timedelta(hours=model_time_shift) for hr in model_hours_tmp]
        for si,stat_key in enumerate(stat_keys):
            # 3D
            tmp = data[G.MODEL][G.STAT][stat_key][G.RAW][lm_run]['tcm']
            for hi,hour in enumerate(model_hours_tmp):
                loc_str = hour.strftime('%Y-%m-%d %H')
                hr_ind = lm_inds[hi]

                tcm[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['tcm'].loc[loc_str].values
                zvp10[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['zvp10'].loc[loc_str].values

            # 2D
            obs_gust[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_GUST_SPEED][model_hours_shifted] 
            obs_mean[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_MEAN_WIND][model_hours_shifted] 


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

    pickle.dump(data, open(CN.train_readj_path, 'wb'))

else:

    data = pickle.load( open(CN.train_readj_path, 'rb') )

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
    learning_rate = 1E-5

    c = 0
    while np.abs(np.mean(d_errors)) > d_error_thresh:


        # SGD selection
        sgd_inds = np.random.choice([True, False], (zvp10.shape[0]), p=[sgd_prob,1-sgd_prob])
        sgd_zvp10 = zvp10[sgd_inds,:]
        sgd_tcm = tcm[sgd_inds,:]
        sgd_obs_gust = obs_gust[sgd_inds]
        N = len(sgd_obs_gust)

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

        # error
        deviation = sgd_obs_gust - sgd_gust_max
        error_now = np.sqrt(np.sum(deviation**2)/N)
        d_error = error_old - error_now
        d_errors = np.roll(d_errors, shift=1)
        d_errors[0] = d_error
        error_old = error_now
        if i_output_error:
            if c % 10 == 0:
                print(str(c) + '\t' + str(error_now) + '\t' + str(np.abs(np.mean(d_errors))))
                print('alpha 1 ' + str(alpha1) + ' alpha 2 ' + str(alpha2))

        # gradient of parameters
        if mode == 'ln':
            dalpha1 = -2/N * np.sum( sgd_tcm_max*sgd_zvp10_max * deviation )
            dalpha2 = 0
        elif mode == 'nl':
            dalpha1 = -2/N * np.sum( sgd_tcm_max*sgd_zvp10_max    * deviation )
            dalpha2 = -2/N * np.sum( sgd_tcm_max*sgd_zvp10_max**2 * deviation )
        else:
            raise ValueError('wrong mode')

        alpha1 = alpha1 - learning_rate * dalpha1
        alpha2 = alpha2 - learning_rate * dalpha2

        # adjust learning rate
        learning_rate = error_now*learning_rate_factor

        c += 1

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

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from functions import plot_error
import globals as G
from namelist_cases import Case_Namelist

############ USER INPUT #############
case_index = 0
CN = Case_Namelist(case_index)
# do not plot (0) show plot (1) save plot (2)
i_plot = 2
model_dt = 10
i_label = ''
i_load = 1
i_output_error = 1
default_learning_rate_factor = 1E-2

modes = ['ln',
         'nl']
i_mode_ints = range(0,len(modes))
i_mode_ints = [0]
max_mean_wind_error = 1.0
i_overwrite_param_file = 0
#####################################

# create directories
if i_plot > 1 and not os.path.exists(CN.plot_path):
    os.mkdir(CN.plot_path)


if not i_load:
    # load data
    data = pickle.load( open(CN.mod_path, 'rb') )

    #stat_keys = ['AEG','ABO', 'AIG']
    #stat_keys = ['ABO', 'AEG']
    stat_keys = data[G.STAT_NAMES]


    lm_runs = list(data[G.MODEL][G.STAT][stat_keys[0]][G.RAW].keys())
    n_hours = len(lm_runs)*24
    n_stats = len(stat_keys)
    ts_per_hour = int(3600/model_dt)


    # 3D
    model_mean = np.full((n_hours, n_stats, ts_per_hour), np.nan)
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
            obs_gust[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_GUST_SPEED][model_hours_tmp] 
            obs_mean[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_MEAN_WIND][model_hours_tmp] 



    # Process fields
    tcm[tcm < 5E-4] = 5E-4
    tcm = np.sqrt(tcm)

        
    # observation to 1D and filter values
    obs_gust_flat = obs_gust.flatten()
    obs_mean_flat = obs_mean.flatten()
    model_mean = np.mean(zvp10, axis=2).flatten()


    data = {}
    data['model_mean'] = model_mean
    data['obs_gust_flat'] = obs_gust_flat
    data['obs_mean_flat'] = obs_mean_flat 
    data['tcm'] = tcm 
    data['zvp10'] = zvp10 

    pickle.dump(data, open(CN.train_readj_path, 'wb'))

else:

    data = pickle.load( open(CN.train_readj_path, 'rb') )

    model_mean = data['model_mean']
    obs_gust_flat = data['obs_gust_flat']
    obs_mean_flat = data['obs_mean_flat']
    tcm = data['tcm']
    zvp10 = data['zvp10']



# obs nan mask
obsmask = np.isnan(obs_gust_flat)
obsmask[np.isnan(obs_mean_flat)] = True
# bad mean wind accuracy mask
mean_abs_error = np.abs(model_mean - obs_mean_flat)
mean_rel_error = mean_abs_error/obs_mean_flat
errormask = mean_rel_error > max_mean_wind_error
# combine both
obsmask[errormask] = True

obs_gust_flat = obs_gust_flat[~obsmask] 
obs_mean_flat = obs_mean_flat[~obsmask]
model_mean = model_mean[~obsmask]

# initial gust
gust = zvp10 + 7.2*tcm*zvp10
#gust_diff = 7.2*tcm*zvp10
gust_max_unscaled = np.amax(gust,axis=2).flatten()[~obsmask]

N = obs_gust_flat.shape[0]

for mode_int in i_mode_ints:
    mode = modes[mode_int]
    print('#################################################################################')
    print('############################## ' + str(mode) + ' ################################')

    if mode == 'nl':
        learning_rate_factor = default_learning_rate_factor * 1/20
    else:
        learning_rate_factor = default_learning_rate_factor


    alpha1 = 0
    alpha2 = 0
    error_old = np.Inf
    d_error = 100
    learning_rate = 1E-5
    d_error_thresh = 1E-4

    c = 0
    while (np.abs(d_error) > d_error_thresh) or (c < 10):

        # calc current time step gusts
        if mode == 'ln':
            gust = zvp10 + alpha1*tcm*zvp10
        elif mode == 'nl':
            gust = zvp10 + alpha1*tcm*zvp10 + alpha2*tcm*zvp10**2
        else:
            raise ValueError('wrong mode')

        # find maximum gust
        maxid = gust.argmax(axis=2)
        I,J = np.indices(maxid.shape)
        #print(maxid[0,0])

        tcm_max = tcm[I,J,maxid].flatten()[~obsmask]
        zvp10_max = zvp10[I,J,maxid].flatten()[~obsmask]
        gust_max = gust[I,J,maxid].flatten()[~obsmask]

        # error
        deviation = obs_gust_flat - gust_max
        #quit()
        #error_now = np.sqrt(np.sum(deviation**2)/N)
        error_now = np.sqrt(np.sum(deviation**2)/N)
        d_error = error_old - error_now
        error_old = error_now
        #print(d_error)
        if i_output_error:
            print(str(c) + '   ' + str(error_now) + '   ' + str(d_error))
            #pass

        # gradient of parameters
        if mode == 'ln':
            dalpha1 = -2/N * np.sum( tcm_max*zvp10_max * deviation )
            dalpha2 = 0
        elif mode == 'nl':
            dalpha1 = -2/N * np.sum( tcm_max*zvp10_max * deviation )
            dalpha2 = -2/N * np.sum( tcm_max*zvp10_max**2 * deviation )
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

    # SAVE PARAMETERS 
    if os.path.exists(CN.params_readj_path):
        params = pickle.load( open(CN.params_readj_path, 'rb') )
    else:
        params = {}
    params[mode] = {'alphas':{'1':alpha1,'2':alpha2}}
    pickle.dump(params, open(CN.params_readj_path, 'wb'))

    # PLOT
    plot_error(obs_gust_flat, model_mean, obs_mean_flat, gust_max, gust_max_unscaled)
    plt.suptitle('ADJUST  '+mode)

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


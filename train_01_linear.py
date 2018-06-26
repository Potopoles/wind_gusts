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
#min_gust = 10
# do not plot (0) show plot (1) save plot (2)
i_plot = 2
model_dt = 10
learning_rate = 1E-2
d_error_thresh = 0.1*learning_rate
i_scaling = 1
i_label = 'new'
#i_label = 'fullscaled'
i_output_error = 1

modes = ['linear_normal',
        'linear_no_tcm',
        'linear_heightInt',             # 2
        'linear_no_tcm_heightInt',
        'linear_height',
        'linear_no_tcm_height',
        'nonlinear_normal',             # 6
        'nonlinear_no_tcm',
        'nonlinear_heightInt',
        'nonlinear_no_tcm_heightInt',
        'nonlinear_height',
        'nonlinear_no_tcm_height']
i_mode_ints = range(0,12)
#i_mode_ints = [0]
#####################################

# create directories
if i_plot > 1 and not os.path.exists(CN.plot_path):
    os.mkdir(CN.plot_path)


for mode_int in i_mode_ints:
    mode = modes[mode_int]
    print('#################################################################################')
    print('############################## ' + str(mode) + ' ################################')

    # load data
    data = pickle.load( open(CN.mod_path, 'rb') )

    #stat_keys = ['AEG','ABO', 'AIG']
    #stat_keys = ['ABO', 'AEG']
    stat_keys = data[G.STAT_NAMES]


    lm_runs = list(data[G.MODEL][G.STAT][stat_keys[0]][G.RAW].keys())
    #lm_runs = [lm_runs[2]]
    n_hours = len(lm_runs)*24
    n_stats = len(stat_keys)
    ts_per_hour = int(3600/model_dt)


    # 3D
    tcm = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    zvp10 = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    height = np.full((n_hours, n_stats, ts_per_hour), np.nan)
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
                height[hr_ind,si,:] =data[G.STAT_META][stat_key]['hsurf'].values 

            # 2D
            obs_gust[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_GUST_SPEED][model_hours_tmp] 
            obs_mean[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_MEAN_WIND][model_hours_tmp] 



    # Process fields
    tcm[tcm < 5E-4] = 5E-4
    tcm = np.sqrt(tcm)

        
    # observation to 1D and filter values
    obs_gust_flat = obs_gust.flatten()
    obs_mean_flat = obs_mean.flatten()
    obsmask = np.isnan(obs_gust_flat)
    obsmask[np.isnan(obs_mean_flat)] = True
    obs_gust_flat = obs_gust_flat[~obsmask] 
    obs_mean_flat = obs_mean_flat[~obsmask]
    N = obs_gust_flat.shape[0]


    # keep unscaled mean wind
    zvp10_unsc = zvp10

    # initial gust
    gust = zvp10_unsc + 7.2*tcm*zvp10
    #gust_diff = 7.2*tcm*zvp10
    gust_max_init = np.amax(gust,axis=2).flatten()[~obsmask]

    ## SCALING
    if i_scaling:
        # mean
        mean_tcm = np.mean(tcm)
        mean_zvp10 = np.mean(zvp10)
        mean_height = np.mean(height)

        tcm = tcm - mean_tcm
        zvp10 = zvp10 - mean_zvp10
        height = height - mean_height

        # standard deviation
        sd_tcm = np.std(tcm)
        sd_zvp10 = np.std(zvp10)
        sd_height = np.std(height)

        tcm = tcm/sd_tcm
        zvp10 = zvp10/sd_zvp10
        height = height/sd_height


    alpha0 = 0
    alpha1 = 0
    alpha2 = 0
    alpha3 = 0
    error_old = np.Inf
    d_error = 100

    while np.abs(d_error) > d_error_thresh:
        # calc current time step gusts
        if mode == 'linear_normal':
            gust_diff = alpha0 + alpha1*tcm*zvp10
        elif mode == 'linear_no_tcm':
            gust_diff = alpha0 + alpha1*zvp10
        elif mode == 'linear_heightInt':
            gust_diff = alpha0 + alpha1*tcm*zvp10 + alpha2*height*tcm*zvp10
        elif mode == 'linear_no_tcm_heightInt':
            gust_diff = alpha0 + alpha1*zvp10 + alpha2*height*zvp10
        elif mode == 'linear_height':
            gust_diff = alpha0 + alpha1*tcm*zvp10 + alpha2*height
        elif mode == 'linear_no_tcm_height':
            gust_diff = alpha0 + alpha1*zvp10 + alpha2*height
        elif mode == 'nonlinear_normal':
            gust_diff = alpha0 + alpha1*tcm*zvp10 + alpha3*tcm*zvp10**2
        elif mode == 'nonlinear_no_tcm':
            gust_diff = alpha0 + alpha1*zvp10 + alpha3*zvp10**2
        elif mode == 'nonlinear_heightInt':
            gust_diff = alpha0 + alpha1*tcm*zvp10 + alpha2*height*tcm*zvp10 + alpha3*tcm*zvp10**2
        elif mode == 'nonlinear_no_tcm_heightInt':
            gust_diff = alpha0 + alpha1*zvp10 + alpha2*height*zvp10 + alpha3*zvp10**2
        elif mode == 'nonlinear_height':
            gust_diff = alpha0 + alpha1*tcm*zvp10 + alpha2*height + alpha3*tcm*zvp10**2
        elif mode == 'nonlinear_no_tcm_height':
            gust_diff = alpha0 + alpha1*zvp10 + alpha2*height + alpha3*zvp10**2

        # calc model gust
        #print(gust_diff[0,0,:10])
        #print(np.mean(gust_diff))
        gust = zvp10_unsc + gust_diff
        #print(gust[0,0,:10])

        # find maximum gust
        maxid = gust.argmax(axis=2)
        I,J = np.indices(maxid.shape)
        #print(maxid[0,0])

        tcm_max = tcm[I,J,maxid].flatten()[~obsmask]
        zvp10_max = zvp10[I,J,maxid].flatten()[~obsmask]
        gust_max = gust[I,J,maxid].flatten()[~obsmask]
        gust_diff_max = gust_diff[I,J,maxid].flatten()[~obsmask]
        height_max = height[I,J,maxid].flatten()[~obsmask]

        # calc obs gust diff
        #obs_gust_diff_max = obs_gust_flat - zvp10_max
        obs_gust_diff_max = obs_gust_flat - obs_mean_flat

        # error
        deviation = obs_gust_diff_max - gust_diff_max
        #quit()
        #error_now = np.sqrt(np.sum(deviation**2)/N)
        error_now = np.sqrt(np.sum(deviation**2)/N)
        d_error = error_old - error_now
        error_old = error_now
        if i_output_error:
            print(error_now)
            #pass

        # gradient of parameters
        if mode == 'linear_normal':
            dalpha0 = -2/N * np.sum(deviation)
            dalpha1 = -2/N * np.sum( tcm_max*zvp10_max * deviation )
            dalpha2 = 0
            dalpha3 = 0
        elif mode == 'linear_no_tcm':
            dalpha0 = -2/N * np.sum(deviation)
            dalpha1 = -2/N * np.sum( zvp10_max * deviation )
            dalpha2 = 0
            dalpha3 = 0
        elif mode == 'linear_heightInt':
            dalpha0 = -2/N * np.sum(deviation)
            dalpha1 = -2/N * np.sum( tcm_max*zvp10_max * deviation )
            dalpha2 = -2/N * np.sum( height_max*tcm_max*zvp10_max * deviation )
            dalpha3 = 0
        elif mode == 'linear_no_tcm_heightInt':
            dalpha0 = -2/N * np.sum(deviation)
            dalpha1 = -2/N * np.sum( zvp10_max * deviation )
            dalpha2 = -2/N * np.sum( height_max*zvp10_max * deviation )
            dalpha3 = 0
        elif mode == 'linear_height':
            dalpha0 = -2/N * np.sum(deviation)
            dalpha1 = -2/N * np.sum( tcm_max*zvp10_max * deviation )
            dalpha2 = -2/N * np.sum( height_max * deviation )
            dalpha3 = 0
        elif mode == 'linear_no_tcm_height':
            dalpha0 = -2/N * np.sum(deviation)
            dalpha1 = -2/N * np.sum( zvp10_max * deviation )
            dalpha2 = -2/N * np.sum( height_max * deviation )
            dalpha3 = 0

        if mode == 'nonlinear_normal':
            dalpha0 = -2/N * np.sum(deviation)
            dalpha1 = -2/N * np.sum( tcm_max*zvp10_max * deviation )
            dalpha2 = 0
            dalpha3 = -2/N * np.sum( tcm_max*zvp10_max**2 * deviation )
        elif mode == 'nonlinear_no_tcm':
            dalpha0 = -2/N * np.sum(deviation)
            dalpha1 = -2/N * np.sum( zvp10_max * deviation )
            dalpha2 = 0
            dalpha3 = -2/N * np.sum( zvp10_max**2 * deviation )
        elif mode == 'nonlinear_heightInt':
            dalpha0 = -2/N * np.sum(deviation)
            dalpha1 = -2/N * np.sum( tcm_max*zvp10_max * deviation )
            dalpha2 = -2/N * np.sum( height_max*tcm_max*zvp10_max * deviation )
            dalpha3 = -2/N * np.sum( tcm_max*zvp10_max**2 * deviation )
        elif mode == 'nonlinear_no_tcm_heightInt':
            dalpha0 = -2/N * np.sum(deviation)
            dalpha1 = -2/N * np.sum( zvp10_max * deviation )
            dalpha2 = -2/N * np.sum( height_max*zvp10_max * deviation )
            dalpha3 = -2/N * np.sum( zvp10_max**2 * deviation )
        elif mode == 'nonlinear_height':
            dalpha0 = -2/N * np.sum(deviation)
            dalpha1 = -2/N * np.sum( tcm_max*zvp10_max * deviation )
            dalpha2 = -2/N * np.sum( height_max * deviation )
            dalpha3 = -2/N * np.sum( tcm_max*zvp10_max**2 * deviation )
        elif mode == 'nonlinear_no_tcm_height':
            dalpha0 = -2/N * np.sum(deviation)
            dalpha1 = -2/N * np.sum( zvp10_max * deviation )
            dalpha2 = -2/N * np.sum( height_max * deviation )
            dalpha3 = -2/N * np.sum( zvp10_max**2 * deviation )

        alpha0 = alpha0 - learning_rate * dalpha0
        alpha1 = alpha1 - learning_rate * dalpha1
        alpha2 = alpha2 - learning_rate * dalpha2
        alpha3 = alpha3 - learning_rate * dalpha3

    print('############')
    print('alpha0 ' + str(alpha0))
    print('alpha1 ' + str(alpha1))
    print('alpha2 ' + str(alpha2))
    print('alpha3 ' + str(alpha3))
    print('############')

    plot_error(obs_gust_flat, gust_max, gust_max_init)
    plt.suptitle(mode)

    if i_plot == 1:
        plt.show()
    elif i_plot > 1:
        if i_label == '':
            plot_name = CN.plot_path + 'tuning_'+str(mode)+'.png'
        else:
            plot_name = CN.plot_path + 'tuning_'+str(i_label)+'_'+str(mode)+'.png'
        print(plot_name)
        plt.savefig(plot_name)
        plt.close('all')


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
i_label = ''
i_output_error = 1

modes = ['k_linear']
#i_mode_ints = range(0,12)
i_mode_ints = [0]
#####################################


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
kval_est = np.full((n_hours, n_stats, ts_per_hour), np.nan)
gust_est = np.full((n_hours, n_stats, ts_per_hour), np.nan)
height = np.full((n_hours, n_stats, ts_per_hour), np.nan)
print('3D shape ' + str(kval_est.shape))
# 2D
obs_gust = np.full((n_hours, n_stats), np.nan)
obs_mean = np.full((n_hours, n_stats), np.nan)

for lmi,lm_run in enumerate(lm_runs):
    print(lm_run)
    lm_inds = np.arange(lmi*24,(lmi+1)*24)
    model_hours_tmp = data[G.MODEL][G.STAT][stat_keys[0]][G.RAW][lm_run]\
                                ['k_bra_es'].resample('H').max().index
    for si,stat_key in enumerate(stat_keys):
        # 3D
        tmp = data[G.MODEL][G.STAT][stat_key][G.RAW][lm_run]['k_bra_es']
        for hi,hour in enumerate(model_hours_tmp):
            loc_str = hour.strftime('%Y-%m-%d %H')
            hr_ind = lm_inds[hi]

            kval_est[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                        [lm_run]['k_bra_es'].loc[loc_str].values
            gust_est[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                        [lm_run]['zv_bra_es'].loc[loc_str].values
            height[hr_ind,si,:] = data[G.STAT_META][stat_key]['hsurf'].values 

        # 2D
        obs_gust[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_GUST_SPEED][model_hours_tmp] 
        obs_mean[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_MEAN_WIND][model_hours_tmp] 



# Process fields
pass

    
# observation to 1D and filter values
obs_gust_flat = obs_gust.flatten()
obs_mean_flat = obs_mean.flatten()
obsmask = np.isnan(obs_gust_flat)
obs_gust_flat = obs_gust_flat[~obsmask] 
N = obs_gust_flat.shape[0]


# find maximum gust
maxid = gust_est.argmax(axis=2)
I,J = np.indices(maxid.shape)

gust_est_max_unscaled = gust_est[I,J,maxid].flatten()[~obsmask]


## SCALING
if i_scaling:
    # mean
    mean_kval_est = np.mean(kval_est)
    mean_gust_est = np.mean(gust_est)
    mean_height = np.mean(height)

    kval_est = kval_est - mean_kval_est
    gust_est = gust_est - mean_gust_est
    height = height - mean_height

    # standard deviation
    sd_kval_est = np.std(kval_est)
    sd_gust_est = np.std(gust_est)
    sd_height = np.std(height)

    kval_est = kval_est/sd_kval_est
    gust_est = gust_est/sd_gust_est
    height = height/sd_height


kval_est_max = kval_est[I,J,maxid].flatten()[~obsmask]
gust_est_max = gust_est[I,J,maxid].flatten()[~obsmask]
height_max = height[I,J,maxid].flatten()[~obsmask]


for mode_int in i_mode_ints:
    mode = modes[mode_int]
    print('#################################################################################')
    print('############################## ' + str(mode) + ' ################################')

    alpha0 = 0
    alpha1 = 0
    alpha2 = 0
    alpha3 = 0
    error_old = np.Inf
    d_error = 100

    while np.abs(d_error) > d_error_thresh:

        # calc current time step gusts
        if mode == 'k_linear':
            gust_max = alpha0 + alpha1*gust_est_max + alpha2*kval_est_max

        # error
        deviation = obs_gust_flat - gust_max
        error_now = np.sqrt(np.sum(deviation**2)/N)
        d_error = error_old - error_now
        error_old = error_now
        if i_output_error:
            print(error_now)
            #pass

        # gradient of parameters
        if mode == 'k_linear':
            dalpha0 = -2/N * np.sum( 1 * deviation )
            dalpha1 = -2/N * np.sum( gust_est_max * deviation )
            dalpha2 = -2/N * np.sum( kval_est_max * deviation )
            dalpha3 = 0

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

    plot_error(obs_gust_flat, gust_max, gust_est_max_unscaled)
    plt.suptitle(mode)
    #plt.scatter(obs_gust_flat, gust_est_max_unscaled)
    #plt.show()
    #quit()

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


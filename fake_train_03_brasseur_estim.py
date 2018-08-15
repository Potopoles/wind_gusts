import numpy as np
import os
import copy
import matplotlib.pyplot as plt
import pickle
from functions import plot_error
import globals as G
from namelist_cases import Case_Namelist
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from functions_train import braes_feature_matrix
from datetime import timedelta

############ USER INPUT #############
case_index = 10
#case_index = 0
CN = Case_Namelist(case_index)
# do not plot (0) show plot (1) save plot (2)
i_plot = 2
model_dt = 10
i_scaling = 1
i_label =  ''
i_load = 1
delete_existing_param_file = 0
modes = ['gust',
        'gust_kheight',
        'gust_height',
        'gust_mean',
        'gust_mean_height',
        'gust_mean_mean2',
        'gust_mean_mean2_height',
        'gust_mean_mean2_mean3',
        'gust_mean_mean2_mean3_height',
        'gust_mean_kheight',
        'gust_mean_height_mean2_kheight']

i_mode_ints = range(0,len(modes))
i_mode_ints = [8]
min_gust = 0
#i_sample_weight = 'linear'
#i_sample_weight = 'squared'
i_sample_weight = '1'
max_mean_wind_error = 100.0
model_time_shift = 1
#####################################

if delete_existing_param_file:
    try:
        os.remove(CN.params_braes_path)
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
    model_mean = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    kval_est = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    gust_est = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    height = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    sso = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    print('3D shape ' + str(kval_est.shape))
    # 2D
    obs_gust = np.full((n_hours, n_stats), np.nan)
    obs_mean = np.full((n_hours, n_stats), np.nan)

    for lmi,lm_run in enumerate(lm_runs):
        print(lm_run)
        lm_inds = np.arange(lmi*24,(lmi+1)*24)
        model_hours_tmp = data[G.MODEL][G.STAT][stat_keys[0]][G.RAW][lm_run]\
                                    ['k_bra_es'].resample('H').max().index
        #print(model_hours_tmp)
        #print(lm_inds)
        model_hours_shifted = [hr+timedelta(hours=model_time_shift) for hr in model_hours_tmp]
        #print(model_hours_shifted)
        for si,stat_key in enumerate(stat_keys):
            # 3D
            tmp = data[G.MODEL][G.STAT][stat_key][G.RAW][lm_run]['k_bra_es']
            for hi,hour in enumerate(model_hours_tmp):
                loc_str = hour.strftime('%Y-%m-%d %H')
                hr_ind = lm_inds[hi]

                model_mean[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['zvp10'].loc[loc_str].values
                kval_est[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['k_bra_es'].loc[loc_str].values
                gust_est[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['zv_bra_es'].loc[loc_str].values
                height[hr_ind,si,:] = data[G.STAT_META][stat_key]['hsurf'].values 

            # 2D
            #obs_gust[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_GUST_SPEED][model_hours_tmp] 
            #obs_mean[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_MEAN_WIND][model_hours_tmp] 
            obs_gust[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_GUST_SPEED][model_hours_shifted] 
            obs_mean[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_MEAN_WIND][model_hours_shifted] 

    #obs_mean = np.nanmean(obs_mean, axis=1)
    #model_mean = np.max(model_mean, axis=2)
    #model_mean = np.mean(model_mean, axis=1)
    #plt.plot(obs_mean)
    #plt.plot(model_mean)
    #plt.show()
    #quit()

    # Process fields
    kheight_est = copy.deepcopy(kval_est)
    kalts = np.loadtxt('../data/kaltitudes.txt')
    kinds = kalts[:,0].astype(np.int)
    kalts = kalts[:,1]
    for i,kind in enumerate(kinds):
        kheight_est[kval_est == kind] = kalts[i]

    data = {}
    data['model_mean'] = model_mean
    data['gust_est'] = gust_est
    data['kheight_est'] = kheight_est
    data['height'] = height
    data['obs_gust'] = obs_gust
    data['obs_mean'] = obs_mean 

    pickle.dump(data, open(CN.train_braes_path, 'wb'))
else:
    data = pickle.load( open(CN.train_braes_path, 'rb') )

    model_mean = data['model_mean']
    gust_est = data['gust_est']
    kheight_est = data['kheight_est']
    height = data['height']
    obs_gust = data['obs_gust']
    obs_mean = data['obs_mean']

# observation to 1D and filter values
obsmask = np.isnan(obs_gust)
obsmask[np.isnan(obs_mean)] = True # TODO NEW
obsmask[obs_gust < min_gust] = True
model_mean_hr = np.mean(model_mean, axis=2)
mean_abs_error = np.abs(model_mean_hr - obs_mean)
mean_rel_error = mean_abs_error/obs_mean
obsmask[mean_rel_error > max_mean_wind_error] = True
obs_gust = obs_gust[~obsmask] 
obs_mean = obs_mean[~obsmask] 
model_mean = model_mean[~obsmask]
model_mean_hr = model_mean_hr[~obsmask]
gust_est = gust_est[~obsmask]
kheight_est = kheight_est[~obsmask]
height = height[~obsmask]
N = obs_gust.flatten().shape[0]

# find maximum gust
maxid = gust_est.argmax(axis=1)
I = np.indices(maxid.shape)
gust_est_max = gust_est[I,maxid].flatten()
gust_est_max_unscaled = gust_est[I,maxid].flatten()
model_mean_max = model_mean[I,maxid].flatten() 
gust_est_max = gust_est[I,maxid].flatten()
kheight_est_max = kheight_est[I,maxid].flatten()
height_max = height[I,maxid].flatten()

regr = LinearRegression(fit_intercept=False)

for mode_int in i_mode_ints:
    mode = modes[mode_int]
    print('#################################################################################')
    print('############################## ' + str(mode) + ' ################################')

    # calc current time step gusts
    #X = braes_feature_matrix(mode, gust_est_max, kheight_est_max,
    #                                height_max, model_mean_max)
    # TODO NEW
    X = braes_feature_matrix(mode, gust_est_max, kheight_est_max,
                                    height_max, obs_mean)
    y = obs_gust

    # scaling
    if i_scaling:
        scaler = StandardScaler(with_mean=False)
        X = scaler.fit_transform(X)

    if i_sample_weight == 'linear':
        regr.fit(X,y, sample_weight=obs_gust)
    elif i_sample_weight == 'squared':
        regr.fit(X,y, sample_weight=obs_gust**2)
    else:
        regr.fit(X,y, sample_weight=np.repeat(1,len(obs_gust)))
 
    alphas = regr.coef_
    print('alphas scaled  ' + str(alphas))
    gust_max = regr.predict(X)

    try:
        #plot_error(obs_gust, model_mean_hr, obs_mean, gust_max, gust_est_max_unscaled)
        # TODO NEW
        plot_error(obs_gust, obs_mean, obs_mean, gust_max, gust_est_max_unscaled)
        plt.suptitle('BRAEST  '+mode)

        if i_plot == 1:
            plt.show()
        elif i_plot > 1:
            if i_label == '':
                # TODO NEW
                plot_name = CN.plot_path + 'fake_tuning_braes_sw_'+i_sample_weight+'_mwa_'+str(max_mean_wind_error)+'_'\
                                            +str(mode)+'.png'
            else:
                # TODO NEW
                plot_name = CN.plot_path + 'fake_tuning_braes_sw_'+i_sample_weight+'_mwa_'+str(max_mean_wind_error)+'_'\
                                            +str(i_label)+'_'+str(mode)+'.png'
            print(plot_name)
            plt.savefig(plot_name)
            plt.close('all')
    except:
        print('Tkinter ERROR while plotting!')

    # RESCALE ALPHA VALUES
    # not necessary to treat powers > 1 different because
    # this is already contained in X matrix
    alphas = alphas/scaler.scale_
    print('alphas unscal  ' + str(alphas))

    # SAVE PARAMETERS 
    if os.path.exists(CN.params_braes_path):
        params = pickle.load( open(CN.params_braes_path, 'rb') )
    else:
        params = {}
    params[mode] = alphas
    pickle.dump(params, open(CN.params_braes_path, 'wb'))

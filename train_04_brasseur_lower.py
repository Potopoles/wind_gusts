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
from functions_train import bralb_feature_matrix
from datetime import timedelta

############ USER INPUT #############
case_index = 0
CN = Case_Namelist(case_index)
# do not plot (0) show plot (1) save plot (2)
i_plot = 1
model_dt = 10
nhrs_forecast = 24
i_scaling = 1
i_label =  ''
i_load = 0
i_train = 0
delete_existing_param_file = 1
modes = ['gust',
         'gust_gust2',
         'gust_kheight',
         'gust_height',
         'gust_mean',
         'gust_mean_mean2',
         'gust_mean_kheight',
         'gust_mean_height',
         'gust_mean_height_mean2_kheight',
         'gust_mean_height_mean2']

i_mode_ints = range(0,len(modes))
#i_mode_ints = [3,9]
#sample_weight = 'linear'
#sample_weight = 'squared'
sample_weight = '1'
max_mean_wind_error = 100.0
#####################################

if delete_existing_param_file:
    try:
        os.remove(CN.params_bralb_path)
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
    model_mean = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    kval_lb = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    gust_lb = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    height = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    sso = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    print('3D shape ' + str(kval_lb.shape))
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

                model_mean[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['zvp10'][ts_inds]
                kval_lb[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['k_bra_lb'][ts_inds]
                gust_lb[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['zv_bra_lb'][ts_inds]
                height[hr_ind,si,:] = data[G.STAT_META][stat_key]['hsurf'].values 

            # OBSERVATION DATA
            full_hr_timestamps = data[G.MODEL][G.STAT][stat_keys[0]][G.RAW][lm_run]\
                                        ['tcm'].resample('H').max().index[1:]
            obs_gust[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_GUST_SPEED][full_hr_timestamps] 
            obs_mean[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_MEAN_WIND][full_hr_timestamps] 
    print()


    # Process fields
    kheight_lb = copy.deepcopy(kval_lb)
    kalts = np.loadtxt('../data/kaltitudes.txt')
    kinds = kalts[:,0].astype(np.int)
    kalts = kalts[:,1]
    for i,kind in enumerate(kinds):
        kheight_lb[kval_lb == kind] = kalts[i]

    data = {}
    data['model_mean'] = model_mean
    data['gust_lb'] = gust_lb
    data['kheight_lb'] = kheight_lb
    data['height'] = height
    data['obs_gust'] = obs_gust
    data['obs_mean'] = obs_mean 

    pickle.dump(data, open(CN.train_bralb_path, 'wb'))
else:
    data = pickle.load( open(CN.train_bralb_path, 'rb') )

    model_mean = data['model_mean']
    gust_lb = data['gust_lb']
    kheight_lb = data['kheight_lb']
    height = data['height']
    obs_gust = data['obs_gust']
    obs_mean = data['obs_mean']


if i_train:

    # observation to 1D and filter values
    obsmask = np.isnan(obs_gust)
    obsmask[np.isnan(obs_mean)] = True
    model_mean_hr = np.mean(model_mean, axis=2)
    mean_abs_error = np.abs(model_mean_hr - obs_mean)
    mean_rel_error = mean_abs_error/obs_mean
    obsmask[mean_rel_error > max_mean_wind_error] = True
    obs_gust = obs_gust[~obsmask] 
    obs_mean = obs_mean[~obsmask] 
    model_mean = model_mean[~obsmask]
    model_mean_hr = model_mean_hr[~obsmask]
    gust_lb = gust_lb[~obsmask]
    kheight_lb = kheight_lb[~obsmask]
    height = height[~obsmask]
    N = obs_gust.flatten().shape[0]

    # find maximum gust
    maxid = gust_lb.argmax(axis=1)
    I = np.indices(maxid.shape)
    gust_lb_max = gust_lb[I,maxid].flatten()
    gust_lb_max_unscaled = gust_lb[I,maxid].flatten()
    model_mean_max = model_mean[I,maxid].flatten() 
    gust_lb_max = gust_lb[I,maxid].flatten()
    kheight_lb_max = kheight_lb[I,maxid].flatten()
    height_max = height[I,maxid].flatten()

    regr = LinearRegression(fit_intercept=False)

    for mode_int in i_mode_ints:
        mode = modes[mode_int]
        print('#################################################################################')
        print('############################## ' + str(mode) + ' ################################')

        # calc current time step gusts
        X = bralb_feature_matrix(mode, gust_lb_max, kheight_lb_max,
                                        height_max, model_mean_max)
        y = obs_gust

        # scaling
        if i_scaling:
            scaler = StandardScaler(with_mean=False)
            X = scaler.fit_transform(X)

        if sample_weight == 'linear':
            regr.fit(X,y, sample_weight=obs_gust)
        elif sample_weight == 'squared':
            regr.fit(X,y, sample_weight=obs_gust**2)
        else:
            regr.fit(X,y, sample_weight=np.repeat(1,len(obs_gust)))
     
        alphas = regr.coef_
        print('alphas scaled  ' + str(alphas))
        gust_max = regr.predict(X)

        try:
            plot_error(obs_gust, model_mean_hr, obs_mean, gust_max, gust_lb_max_unscaled)
            plt.suptitle('BRALB  '+mode)

            if i_plot == 1:
                plt.show()
            elif i_plot > 1:
                if i_label == '':
                    plot_name = CN.plot_path + 'tuning_bralb_sw_'+sample_weight+'_mwa_'+str(max_mean_wind_error)+'_'\
                                                +str(mode)+'.png'
                else:
                    plot_name = CN.plot_path + 'tuning_bralb_sw_'+sample_weight+'_mwa_'+str(max_mean_wind_error)+'_'\
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
        if os.path.exists(CN.params_bralb_path):
            params = pickle.load( open(CN.params_bralb_path, 'rb') )
        else:
            params = {}
        params[mode] = alphas
        print(CN.params_bralb_path)
        pickle.dump(params, open(CN.params_bralb_path, 'wb'))


else:
    print('Train is turned off. Finish.')

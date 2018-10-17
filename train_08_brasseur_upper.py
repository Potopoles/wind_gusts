import numpy as np
import os
import copy
import matplotlib.pyplot as plt
import pickle
from functions import plot_error
import globals as G
from namelist_cases import Case_Namelist
import namelist_cases as nl
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from functions_train import braes_feature_matrix
from datetime import timedelta

############ USER INPUT #############
case_index = nl.case_index
CN = Case_Namelist(case_index)
# do not plot (0) show plot (1) save plot (2)
i_plot = nl.i_plot
model_dt = nl.model_dt
nhrs_forecast = nl.nhrs_forecast
i_scaling = 1
i_label =  ''
i_load = nl.i_load
i_train = nl.i_train
delete_existing_param_file = nl.delete_existing_param_file
#max_mean_wind_error = nl.max_mean_wind_error
#sample_weight = nl.sample_weight

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
#i_mode_ints = [3]
#####################################

if delete_existing_param_file:
    try:
        os.remove(CN.params_braub_path)
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
    kval = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    gust_bra = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    height = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    sso = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    print('3D shape ' + str(kval.shape))
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
                kval[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['k_bra_ub'][ts_inds]
                gust_bra[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['zv_bra_ub'][ts_inds]
                height[hr_ind,si,:] = data[G.STAT_META][stat_key]['hsurf'].values 

            # OBSERVATION DATA
            full_hr_timestamps = data[G.MODEL][G.STAT][stat_keys[0]][G.RAW][lm_run]\
                                        ['tcm'].resample('H').max().index[1:]
            obs_gust[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_GUST_SPEED][full_hr_timestamps] 
            obs_mean[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_MEAN_WIND][full_hr_timestamps] 
    print()

    #obs_mean = np.nanmean(obs_mean, axis=1)
    #model_mean = np.max(model_mean, axis=2)
    #model_mean = np.mean(model_mean, axis=1)
    #plt.plot(obs_mean)
    #plt.plot(model_mean)
    #plt.show()
    #quit()

    # Process fields
    kheight = copy.deepcopy(kval)
    kalts = np.loadtxt('../data/kaltitudes.txt')
    kinds = kalts[:,0].astype(np.int)
    kalts = kalts[:,1]
    for i,kind in enumerate(kinds):
        kheight[kval == kind] = kalts[i]

    data = {}
    data['model_mean'] = model_mean
    data['gust_bra'] = gust_bra
    data['kheight'] = kheight
    data['height'] = height
    data['obs_gust'] = obs_gust
    data['obs_mean'] = obs_mean 

    pickle.dump(data, open(CN.train_braub_path, 'wb'))
else:
    data = pickle.load( open(CN.train_braub_path, 'rb') )

    model_mean = data['model_mean']
    gust_bra = data['gust_bra']
    kheight = data['kheight']
    height = data['height']
    obs_gust = data['obs_gust']
    obs_mean = data['obs_mean']



if i_train:


    # observation to 1D and filter values
    obsmask = np.isnan(obs_gust)
    obsmask[np.isnan(obs_mean)] = True # TODO NEW

    model_mean_hr = np.mean(model_mean, axis=2)

    ## bad mean wind accuracy mask
    #mean_abs_error = np.abs(model_mean_hr - obs_mean)
    #mean_rel_error = mean_abs_error/obs_mean
    #obsmask[mean_rel_error > max_mean_wind_error] = True

    obs_gust = obs_gust[~obsmask] 
    obs_mean = obs_mean[~obsmask] 
    model_mean = model_mean[~obsmask]
    model_mean_hr = model_mean_hr[~obsmask]
    gust_bra = gust_bra[~obsmask]
    kheight = kheight[~obsmask]
    height = height[~obsmask]
    N = obs_gust.flatten().shape[0]
    print(N)

    # find maximum gust
    maxid = gust_bra.argmax(axis=1)
    I = np.indices(maxid.shape)
    gust_bra_max = gust_bra[I,maxid].flatten()
    gust_bra_max_unscaled = gust_bra[I,maxid].flatten()
    model_mean_max = model_mean[I,maxid].flatten() 
    gust_bra_max = gust_bra[I,maxid].flatten()
    kheight_max = kheight[I,maxid].flatten()
    height_max = height[I,maxid].flatten()

    regr = LinearRegression(fit_intercept=False)

    for mode_int in i_mode_ints:
        mode = modes[mode_int]
        print('#################################################################################')
        print('############################## ' + str(mode) + ' ################################')

        # calc current time step gusts
        X = braes_feature_matrix(mode, gust_bra_max, kheight_max,
                                        height_max, model_mean_max)
        y = obs_gust

        # scaling
        if i_scaling:
            scaler = StandardScaler(with_mean=False)
            X = scaler.fit_transform(X)

        #if sample_weight == 'linear':
        #    regr.fit(X,y, sample_weight=obs_gust)
        #elif sample_weight == 'squared':
        #    regr.fit(X,y, sample_weight=obs_gust**2)
        #else:
        #    regr.fit(X,y, sample_weight=np.repeat(1,len(obs_gust)))
        regr.fit(X,y, sample_weight=np.repeat(1,len(obs_gust)))
     
        alphas = regr.coef_
        print('alphas scaled  ' + str(alphas))
        gust_max = regr.predict(X)

        if i_plot > 0:
            try:
                plot_error(obs_gust, model_mean_hr, obs_mean, gust_max, gust_bra_max_unscaled)
                plt.suptitle('BRAUB  '+mode)

                if i_plot == 1:
                    plt.show()
                elif i_plot > 1:
                    if i_label == '':
                        plot_name = CN.plot_path + 'tuning_braub_'+str(mode)+'.png'
                    else:
                        plot_name = CN.plot_path + 'tuning_braub_'+str(i_label)+'_'+str(mode)+'.png'
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
        if os.path.exists(CN.params_braub_path):
            params = pickle.load( open(CN.params_braub_path, 'rb') )
        else:
            params = {}
        params[mode] = alphas
        pickle.dump(params, open(CN.params_braub_path, 'wb'))


else:
    print('Train is turned off. Finish.')
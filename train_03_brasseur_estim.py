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

############ USER INPUT #############
case_index = 1
CN = Case_Namelist(case_index)
# do not plot (0) show plot (1) save plot (2)
i_plot = 2
model_dt = 10
i_scaling = 1
i_label =  ''
i_load = 1
delete_existing_param_file = 1
modes = ['gust',
        'gust_kheight',
        'gust_height',
        'gust_mean',
        'gust_mean_height',
        'gust_mean_mean2',
        'gust_mean_mean2_height',
        'gust_mean_mean2_mean3',
        'gust_mean_mean2_mean3_height']
        'gust_mean_kheight',
        'gust_mean_height_mean2_kheight']

i_mode_ints = range(0,len(modes))
i_mode_ints = [3,5]
min_gust = 0
#i_sample_weight = 'linear'
#i_sample_weight = 'squared'
i_sample_weight = '1'
max_mean_wind_error = 1.0
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
            obs_gust[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_GUST_SPEED][model_hours_tmp] 
            obs_mean[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_MEAN_WIND][model_hours_tmp] 



    # Process fields
    kheight_est = copy.deepcopy(kval_est)
    kalts = np.loadtxt('../data/kaltitudes.txt')
    kinds = kalts[:,0].astype(np.int)
    kalts = kalts[:,1]
    for i,kind in enumerate(kinds):
        kheight_est[kval_est == kind] = kalts[i]

        
    # observation to 1D and filter values
    obs_gust_flat = obs_gust.flatten()
    obs_mean_flat = obs_mean.flatten()
    obsmask = np.isnan(obs_gust_flat)
    obsmask[obs_gust_flat < min_gust] = True
    obs_gust_flat = obs_gust_flat[~obsmask] 
    obs_mean_flat = obs_mean_flat[~obsmask] 
    N = obs_gust_flat.shape[0]

    # find maximum gust
    maxid = gust_est.argmax(axis=2)
    I,J = np.indices(maxid.shape)

    gust_est_max_unscaled = gust_est[I,J,maxid].flatten()[~obsmask]

    model_mean_max = model_mean[I,J,maxid].flatten()[~obsmask] 
    model_mean = np.mean(model_mean, axis=2).flatten()[~obsmask]
    gust_est_max = gust_est[I,J,maxid].flatten()[~obsmask]
    kheight_est_max = kheight_est[I,J,maxid].flatten()[~obsmask]
    height_max = height[I,J,maxid].flatten()[~obsmask]

    data = {}
    data['model_mean_max'] = model_mean_max
    data['model_mean'] = model_mean
    data['gust_est_max'] = gust_est_max
    data['kheight_est_max'] = kheight_est_max
    data['height_max'] = height_max
    data['obs_gust_flat'] = obs_gust_flat
    data['obs_mean_flat'] = obs_mean_flat 
    data['gust_est_max_unscaled'] = gust_est_max_unscaled

    pickle.dump(data, open(CN.train_braes_path, 'wb'))
else:
    data = pickle.load( open(CN.train_braes_path, 'rb') )

    model_mean_max = data['model_mean_max']
    model_mean = data['model_mean']
    gust_est_max = data['gust_est_max']
    kheight_est_max = data['kheight_est_max']
    height_max = data['height_max']
    obs_gust_flat = data['obs_gust_flat']
    obs_mean_flat = data['obs_mean_flat']
    gust_est_max_unscaled = data['gust_est_max_unscaled']

mean_abs_error = np.abs(model_mean - obs_mean_flat)
mean_rel_error = mean_abs_error/obs_mean_flat
errormask = mean_rel_error > max_mean_wind_error

model_mean_max = model_mean_max[~errormask]
model_mean = model_mean[~errormask]
gust_est_max = gust_est_max[~errormask]
kheight_est_max = kheight_est_max[~errormask]
height_max = height_max[~errormask]
obs_gust_flat = obs_gust_flat[~errormask]
obs_mean_flat = obs_mean_flat[~errormask]
gust_est_max_unscaled = gust_est_max_unscaled[~errormask]

regr = LinearRegression(fit_intercept=False)

for mode_int in i_mode_ints:
    mode = modes[mode_int]
    print('#################################################################################')
    print('############################## ' + str(mode) + ' ################################')

    # calc current time step gusts
    X = braes_feature_matrix(mode, gust_est_max, kheight_est_max,
                                    height_max, model_mean_max)
    y = obs_gust_flat

    # scaling
    if i_scaling:
        scaler = StandardScaler(with_mean=False)
        X = scaler.fit_transform(X)

    if i_sample_weight == 'linear':
        regr.fit(X,y, sample_weight=obs_gust_flat)
    elif i_sample_weight == 'squared':
        regr.fit(X,y, sample_weight=obs_gust_flat**2)
    else:
        regr.fit(X,y, sample_weight=np.repeat(1,len(obs_gust_flat)))
 
    alphas = regr.coef_
    print('alphas scaled  ' + str(alphas))
    gust_max = regr.predict(X)

    try:
        plot_error(obs_gust_flat, model_mean, obs_mean_flat, gust_max, gust_est_max_unscaled)
        plt.suptitle('BRAEST  '+mode)

        if i_plot == 1:
            plt.show()
        elif i_plot > 1:
            if i_label == '':
                plot_name = CN.plot_path + 'tuning_braes_sw_'+i_sample_weight+'_mwa_'+str(max_mean_wind_error)+'_'\
                                            +str(mode)+'.png'
            else:
                plot_name = CN.plot_path + 'tuning_braes_sw_'+i_sample_weight+'_mwa_'+str(max_mean_wind_error)+'_'\
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

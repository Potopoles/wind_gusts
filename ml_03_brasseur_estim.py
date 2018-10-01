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
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
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

min_gust = 0

model_time_shift = 1

i_fake_train = 0
#####################################

#if delete_existing_param_file:
#    try:
#        os.remove(CN.params_braes_path)
#    except:
#        pass


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
    kval_est = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    gust_est = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    height = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    sso = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    z0 = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    print('3D shape ' + str(kval_est.shape))
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
                kval_est[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['k_bra_es'][ts_inds]
                gust_est[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['zv_bra_es'][ts_inds]
                height[hr_ind,si,:] = data[G.STAT_META][stat_key]['hsurf'].values 
                sso[hr_ind,si,:] = data[G.STAT_META][stat_key]['sso_stdh'].values 
                z0[hr_ind,si,:] = data[G.STAT_META][stat_key]['z0'].values 

            # OBSERVATION DATA
            full_hr_timestamps = data[G.MODEL][G.STAT][stat_keys[0]][G.RAW][lm_run]\
                                        ['tcm'].resample('H').max().index[1:]
            obs_gust[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_GUST_SPEED][full_hr_timestamps] 
            obs_mean[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_MEAN_WIND][full_hr_timestamps] 

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
    data['sso'] = sso
    data['z0'] = z0
    data['obs_gust'] = obs_gust
    data['obs_mean'] = obs_mean 

    pickle.dump(data, open(CN.ML_braes_path, 'wb'))
else:
    data = pickle.load( open(CN.ML_braes_path, 'rb') )

    model_mean = data['model_mean']
    gust_est = data['gust_est']
    kheight_est = data['kheight_est']
    height = data['height']
    sso = data['sso']
    z0 = data['z0']
    obs_gust = data['obs_gust']
    obs_mean = data['obs_mean']


# observation to 1D and filter values
obsmask = np.isnan(obs_gust)
obsmask[obs_gust < min_gust] = True
obsmask[np.isnan(obs_mean)] = True
model_mean_hr = np.mean(model_mean, axis=2)
obs_gust = obs_gust[~obsmask] 
obs_mean = obs_mean[~obsmask] 
model_mean = model_mean[~obsmask]
model_mean_hr = model_mean_hr[~obsmask]
gust_est = gust_est[~obsmask]
kheight_est = kheight_est[~obsmask]
height = height[~obsmask]
sso = sso[~obsmask]
z0 = z0[~obsmask]
N = obs_gust.flatten().shape[0]

# find maximum gust
maxid = gust_est.argmax(axis=1)
I = np.indices(maxid.shape)
gust_est_max = gust_est[I,maxid].flatten()
gust_est_max_unscaled = gust_est[I,maxid].flatten()
model_mean_max = model_mean[I,maxid].flatten() 
kheight_est_max = kheight_est[I,maxid].flatten()
height_max = height[I,maxid].flatten()
sso_max = sso[I,maxid].flatten()
z0_max = z0[I,maxid].flatten()

if i_fake_train:
    # fake train TODO NEW
    model_mean_max = obs_mean

nfeat = 6
X = np.zeros((gust_est_max.shape[0], nfeat))
X[:,0] = gust_est_max
X[:,1] = model_mean_max
X[:,2] = kheight_est_max
X[:,3] = height_max
X[:,4] = sso_max
X[:,5] = z0_max

poly = PolynomialFeatures(degree=5)
X = poly.fit_transform(X)

if i_scaling:
    scaler = StandardScaler(with_mean=False)
    X = scaler.fit_transform(X)

regr = LinearRegression(fit_intercept=False)
#regr = Lasso(alpha=0.25, fit_intercept=False)

y = obs_gust
regr.fit(X,y, sample_weight=obs_gust**2)

alphas = regr.coef_
print('alphas scaled  ' + str(alphas[np.abs(alphas) > 0.0001]))
gust_max = regr.predict(X)

try:
    if i_fake_train:
        plot_error(obs_gust, obs_mean, obs_mean, gust_max, gust_est_max_unscaled)
    else:
        plot_error(obs_gust, model_mean_hr, obs_mean, gust_max, gust_est_max_unscaled)
    plt.suptitle('BRAEST ML')

    if i_plot == 1:
        plt.show()
    elif i_plot > 1:
        if i_label == '':
            plot_name = CN.plot_path + 'ML_braes_sw_'+i_sample_weight+'_mwa_'+str(max_mean_wind_error)+'.png'
        else:
            plot_name = CN.plot_path + 'ML_braes_sw_'+i_sample_weight+'_mwa_'+str(max_mean_wind_error)+'_'\
                                        +str(i_label)+'.png'
        print(plot_name)
        plt.savefig(plot_name)
        plt.close('all')
except:
    print('Tkinter ERROR while plotting!')

# RESCALE ALPHA VALUES
# not necessary to treat powers > 1 different because
# this is already contained in X matrix
alphas = alphas/scaler.scale_
#print('alphas unscal  ' + str(alphas))

## SAVE PARAMETERS 
#if os.path.exists(CN.params_braes_path):
#    params = pickle.load( open(CN.params_braes_path, 'rb') )
#else:
#    params = {}
#params[mode] = alphas
#pickle.dump(params, open(CN.params_braes_path, 'wb'))

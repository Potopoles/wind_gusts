import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from functions import plot_error, apply_scaling
import globals as G
from namelist_cases import Case_Namelist
import namelist_cases as nl
from functions_train import stat_calculate_gust, stat_combine_features
from datetime import timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
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
learning_rate_factor = 1E-3
d_error_thresh = 1E-5
delete_existing_param_file = nl.delete_existing_param_file
i_scaling = 1
#max_mean_wind_error = nl.max_mean_wind_error
#sample_weight = nl.sample_weight

mode = 'ML'

#feature_names = ['zvp10', 'tcm', 'tkel1', 'hsurf', 'sso_stdh', 'zv_bra_es', 'k_bra_es', 'dvl3v10', 'z0', \
#                'icon_gust']
feature_names = ['zvp10', 'tcm', 'tkel1', 'hsurf', 'zv_bra_es', 'zbra', 'dvl3v10', \
                'uvl3', 'uvl2', 'uvl1', 'ul1', 'vl1', 'z0', 'Tl1', 'shflx', 'qvflx', 'Tskin', 'qvl1', \
                'rhol1', 'phil1', 'ps']
#####################################

# create directories
if i_plot > 1 and not os.path.exists(CN.plot_path):
    os.mkdir(CN.plot_path)

if delete_existing_param_file:
    try:
        os.remove(CN.params_stat_path)
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
    features = {}
    for feat in feature_names:
        features[feat] = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    # 2D
    obs_gust = np.full((n_hours, n_stats), np.nan)
    obs_mean = np.full((n_hours, n_stats), np.nan)

    # level_alts gives an altitude (value) to a model level (key)
    level_altitudes_file = np.loadtxt('../data/kaltitudes.txt')
    level_alts = {}
    for i,line in enumerate(level_altitudes_file[:-1]):
        level_alts[int(line[0])] = (level_altitudes_file[i,1] + level_altitudes_file[i+1,1])/2

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

                for feat in feature_names:
                    if feat in ['zvp10', 'tcm', 'tkel1', 'zv_bra_es', 'uvl3', 'uvl2', 'uvl1', 'ul1', 'vl1', \
                                'tkel1', 'Tl1', 'shflx', 'qvflx', 'Tskin', 'qvl1', 'rhol1', 'phil1', 'ps']:
                        features[feat][hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                                    [lm_run][feat][ts_inds]
                    elif feat in ['hsurf', 'sso_stdh', 'z0']:
                        features[feat][hr_ind,si,:] = data[G.STAT_META][stat_key][feat].values 
                    elif feat == 'dvl3v10':
                        features[feat][hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                                    [lm_run]['uvl3'][ts_inds] - \
                                                      data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                                    [lm_run]['zvp10'][ts_inds]
                    elif feat == 'zbra':
                        features[feat][hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                                    [lm_run]['k_bra_es'][ts_inds]
                        for ind,level in enumerate(features[feat][hr_ind,si,:]):
                            features[feat][hr_ind,si,:][ind] = level_alts[int(level)]
                    elif feat == 'icon_gust':
                        ugn = 7.71
                        hpbl = 1000
                        g = 9.80665
                        Rd = 287.05
                        etv = 0.6078
                        cp = 1005.0
                        kappa = Rd/cp

                        umlev = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                                    [lm_run]['ul1'][ts_inds] 
                        vmlev = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                                    [lm_run]['vl1'][ts_inds] 
                        ps = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                                    [lm_run]['ps'][ts_inds] 
                        qvflx = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                                    [lm_run]['qvflx'][ts_inds] 
                        shflx = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                                    [lm_run]['shflx'][ts_inds] 
                        z0 = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                                    [lm_run]['z0'][ts_inds] 
                        Tskin = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                                    [lm_run]['Tskin'][ts_inds] 
                        Tmlev = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                                    [lm_run]['Tl1'][ts_inds] 
                        qvmlev = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                                    [lm_run]['qvl1'][ts_inds] 
                        phimlev = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                                    [lm_run]['phil1'][ts_inds] 
                        zvp10 = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                                    [lm_run]['zvp10'][ts_inds] 

                        # density
                        rho = ps / ( Rd*Tmlev * ( 1 + etv*qvmlev ) )

                        # buoyancy
                        buoy = g * ( - etv*qvflx - shflx/( Tskin*cp ) ) / rho

                        # surface stress
                        zcdn = ( kappa / np.log( 1 + phimlev/(g*z0) ) )**2
                        dua = np.sqrt( np.maximum( 0.1**2, umlev**2 + vmlev**2 ) )
                        ustr = rho*umlev*dua*zcdn
                        vstr = rho*vmlev*dua*zcdn

                        # friction velocity
                        ustar2 = np.sqrt( ustr**2 + vstr**2 ) / rho 
                        wstar2 = ( buoy[buoy > 0]*hpbl )**(2/3)
                        ustar2[buoy > 0] = ustar2[buoy > 0] + 2E-3*wstar2
                        ustar = np.maximum( np.sqrt(ustar2), 0.0001 )

                        # wind gust
                        idl = -hpbl*kappa*buoy/ustar**3
                        gust = zvp10
                        greater0 = idl >= 0
                        gust[greater0] = gust[greater0] + ustar[greater0]*ugn
                        smaller0 = idl < 0
                        gust[smaller0] = gust[smaller0] + ustar[smaller0]*ugn * (1 - 0.5/12*idl[smaller0])**(1/3)
                        features[feat][hr_ind,si,:] = gust

            # OBSERVATION DATA
            full_hr_timestamps = data[G.MODEL][G.STAT][stat_keys[0]][G.RAW][lm_run]\
                                        ['tcm'].resample('H').max().index[1:]
            obs_gust[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_GUST_SPEED][full_hr_timestamps] 
            obs_mean[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_MEAN_WIND][full_hr_timestamps] 
    print()


    # Process fields
    if 'tcm' in feature_names:
        features['tcm'][features['tcm'] < 5E-4] = 5E-4
        features['tcm'] = np.sqrt(features['tcm'])
        
    model_mean = np.mean(features['zvp10'], axis=2)

    data = {}
    data['model_mean'] = model_mean
    data['obs_gust'] = obs_gust
    data['obs_mean'] = obs_mean
    data['features'] = features
    data['feature_names'] = feature_names

    pickle.dump(data, open(CN.train_stat_path, 'wb'))

else:

    data = pickle.load( open(CN.train_stat_path, 'rb') )

    model_mean = data['model_mean']
    obs_gust = data['obs_gust']
    obs_mean = data['obs_mean']
    features = data['features']
    feature_names = data['feature_names']


# observation to 1D and filter values
obsmask = np.isnan(obs_gust)
obsmask[np.isnan(obs_mean)] = True
obs_gust = obs_gust[~obsmask] 
obs_mean = obs_mean[~obsmask] 
model_mean = model_mean[~obsmask]
for feature in feature_names:
    features[feature] = features[feature][~obsmask]
N = obs_gust.flatten().shape[0]


# find maximum gust
maxid = features['zvp10'].argmax(axis=1)
I = np.indices(maxid.shape)
for feature in feature_names:
    features[feature] = features[feature][I,maxid].flatten()

# TODO DEBUG
#features['zvp10'] = obs_mean

print(features.keys())


if i_train:

    nfeat = 21
    X = np.zeros((obs_gust.shape[0], nfeat))
    X[:,0] = features['zvp10']
    X[:,1] = features['tcm'] 
    X[:,2] = features['tkel1'] 
    X[:,3] = features['hsurf'] 
    X[:,4] = features['zv_bra_es'] 
    X[:,5] = features['zbra'] 
    X[:,6] = features['dvl3v10'] 
    X[:,7] = features['uvl1'] 
    X[:,8] = features['uvl2'] 
    X[:,9] = features['uvl3'] 
    X[:,10] = features['ul1'] 
    X[:,11] = features['vl1'] 
    X[:,12] = features['z0'] 
    X[:,13] = features['Tl1'] 
    X[:,14] = features['shflx'] 
    X[:,15] = features['qvflx'] 
    X[:,16] = features['Tskin'] 
    X[:,17] = features['qvl1'] 
    X[:,18] = features['rhol1'] 
    X[:,19] = features['phil1'] 
    X[:,20] = features['ps'] 


    poly = PolynomialFeatures(degree=2, interaction_only=False)
    X = poly.fit_transform(X)
    print(X)

    if i_scaling:
        scaler = StandardScaler(with_mean=False)
        X = scaler.fit_transform(X)

    regr = LinearRegression(fit_intercept=False)
    #regr = Lasso(alpha=0.25, fit_intercept=False)

    y = obs_gust
    regr.fit(X,y, sample_weight=obs_gust**0)

    alphas = regr.coef_
    print('alphas scaled  ' + str(alphas[np.abs(alphas) > 0.0001]))
    gust_max = regr.predict(X)

    # original gust
    gust_max_unscaled = features['zvp10'] + 7.2*features['tcm'] * features['zvp10']

    if i_plot > 0:
        #try:
        plot_error(obs_gust, model_mean, obs_mean, gust_max, gust_max_unscaled)
        plt.suptitle('STAT  '+mode)

        if i_plot == 1:
            plt.show()
        elif i_plot > 1:
            if i_label == '':
                plot_name = CN.plot_path + 'tuning_stat_'+str(mode)+'.png'
            else:
                plot_name = CN.plot_path + 'tuning_stat_'+str(i_label)+'_'+str(mode)+'.png'
            print(plot_name)
            plt.savefig(plot_name)
            plt.close('all')
        #except:
        #    print('Tkinter ERROR while plotting!')


        ## RESCALE ALPHA VALUES
        #for key,val in alphas.items():
        #    if key in trained:
        #        alphas[key] = val/features_scale[trained[key]['feat']]**trained[key]['power']


        ## SAVE PARAMETERS 
        #if os.path.exists(CN.params_stat_path):# and (i_overwrite_param_file == 0):
        #    params = pickle.load( open(CN.params_stat_path, 'rb') )
        #else:
        #    params = {}
        #params[mode] = alphas
        #pickle.dump(params, open(CN.params_stat_path, 'wb'))

else:
    print('Train is turned off. Finish.')
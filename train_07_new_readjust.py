import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from functions import plot_error
import globals as G
from namelist_cases import Case_Namelist
import namelist_cases as nl
from datetime import timedelta

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

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
default_learning_rate_factor = 1E-2
delete_existing_param_file = nl.delete_existing_param_file
#max_mean_wind_error = nl.max_mean_wind_error
mode = 'mixcoef_alpha00'
#####################################

# create directories
if i_plot > 1 and not os.path.exists(CN.plot_path):
    os.mkdir(CN.plot_path)

if delete_existing_param_file:
    try:
        os.remove(CN.params_readjNEW_path)
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
    tcm = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    zvp10 = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    vbra = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    zbra = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    dvl3v10 = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    hsurf = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    sso_stdh = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    print('3D shape ' + str(tcm.shape))
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

                tcm[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['tcm'][ts_inds]
                zvp10[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['zvp10'][ts_inds]
                vbra[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['zv_bra_lb'][ts_inds]

                zbra[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['k_bra_lb'][ts_inds]
                for ind,level in enumerate(zbra[hr_ind,si,:]):
                    zbra[hr_ind,si,:][ind] = level_alts[int(level)]

                dvl3v10[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['uvl3'][ts_inds] - \
                                              data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['zvp10'][ts_inds]

                hsurf[hr_ind,si,:] = data[G.STAT_META][stat_key]['hsurf'].values 
                sso_stdh[hr_ind,si,:] = data[G.STAT_META][stat_key]['sso_stdh'].values 

            # OBSERVATION DATA
            full_hr_timestamps = data[G.MODEL][G.STAT][stat_keys[0]][G.RAW][lm_run]\
                                        ['tcm'].resample('H').max().index[1:]
            obs_gust[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_GUST_SPEED][full_hr_timestamps] 
            obs_mean[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_MEAN_WIND][full_hr_timestamps] 
    print()


    print(zbra)
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
    data['vbra'] = vbra 
    data['zbra'] = zbra 
    data['dvl3v10'] = dvl3v10 
    data['hsurf'] = hsurf 
    data['sso_stdh'] = sso_stdh 

    data['stations'] = stat_keys 

    pickle.dump(data, open(CN.train_readjNEW_path, 'wb'))

else:

    data = pickle.load( open(CN.train_readjNEW_path, 'rb') )

    model_mean = data['model_mean']
    obs_gust = data['obs_gust']
    obs_mean = data['obs_mean']
    tcm = data['tcm']
    zvp10 = data['zvp10']
    vbra = data['vbra']
    zbra = data['zbra']
    dvl3v10 = data['dvl3v10']
    hsurf = data['hsurf']
    sso_stdh = data['sso_stdh']


#########################
## TEST CASE TO COMPARE WITH test_gusts.py script
#si = 0 # 0: ABO
#gust = zvp10[:,si,:] + 7.2 * tcm[:,si,:] * zvp10[:,si,:]
#gust = np.max(gust,axis=1)
#print(np.round(gust,6))
#quit()
#########################


if i_train:

    # obs nan mask
    obsmask = np.isnan(obs_gust)
    obsmask[np.isnan(obs_mean)] = True
    ## bad mean wind accuracy mask
    #mean_abs_error = np.abs(model_mean - obs_mean)
    #mean_rel_error = mean_abs_error/obs_mean
    #errormask = mean_rel_error > max_mean_wind_error
    ## combine both
    #obsmask[errormask] = True

    obs_gust = obs_gust[~obsmask] 
    obs_mean = obs_mean[~obsmask]
    model_mean = model_mean[~obsmask]
    tcm = tcm[~obsmask]
    zvp10 = zvp10[~obsmask]
    vbra = vbra[~obsmask]
    zbra = zbra[~obsmask]
    dvl3v10 = dvl3v10[~obsmask]
    hsurf = hsurf[~obsmask]
    sso_stdh = sso_stdh[~obsmask]


    # initial gust
    gust = zvp10 + 7.2*tcm*zvp10
    gust_max_orig = np.amax(gust,axis=1)

    maxid = (zvp10).argmax(axis=1)
    I = np.indices(maxid.shape)
    tcm_max = tcm[I,maxid]
    zvp10_max = zvp10[I,maxid]
    vbra_max = vbra[I,maxid]
    zbra_max = zbra[I,maxid]
    dvl3v10_max = dvl3v10[I,maxid]
    hsurf_max = hsurf[I,maxid]
    sso_stdh_max = sso_stdh[I,maxid]


    l_final_gust_timestepwise = True
    i_scaling = 0
    weight_power = 8
    poly_degree = 1
    poly_interaction_only = True

    features = []
    features.append('tcm*zvp10')
    #features.append('zvp10')
    #features.append('vbra')
    #features.append('zbra')
    #features.append('dvl3v10')
    #features.append('hsurf')
    #features.append('sso_stdh')


    X_max = np.zeros( (tcm_max.shape[1],len(features)) )
    X = np.zeros( (tcm.shape[0],tcm.shape[1],len(features)) )
    for fi,feat in enumerate(features):
        if feat == 'tcm*zvp10':
            X_max[:,fi] = zvp10_max[0,:] * tcm_max[0,:]
            X[:,:,fi]   = zvp10          * tcm
        elif feat == 'zvp10':
            X_max[:,fi] = zvp10_max[0,:]
            X[:,:,fi]   = zvp10
        elif feat == 'tcm':
            X_max[:,fi] = tcm_max[0,:]
            X[:,:,fi]   = tcm
        elif feat == 'vbra':
            X_max[:,fi] = vbra_max[0,:]
            X[:,:,fi]   = vbra
        elif feat == 'zbra':
            X_max[:,fi] = zbra_max[0,:]
            X[:,:,fi]   = zbra
        elif feat == 'dvl3v10':
            X_max[:,fi] = dvl3v10_max[0,:]
            X[:,:,fi]   = dvl3v10
        elif feat == 'hsurf':
            X_max[:,fi] = hsurf_max[0,:]
            X[:,:,fi]   = hsurf
        elif feat == 'sso_stdh':
            X_max[:,fi] = sso_stdh_max[0,:]
            X[:,:,fi]   = sso_stdh

    y = obs_gust - zvp10_max[0,:]


    print(X_max)
    print()

    poly = PolynomialFeatures(degree=poly_degree, interaction_only=poly_interaction_only)
    X_max = poly.fit_transform(X_max)
    X_max = X_max[:,1:]

    print(X_max)

    if i_scaling:
        scaler = StandardScaler(with_mean=False)
        X_max = scaler.fit_transform(X_max)

    weights = obs_gust**weight_power

    regr = LinearRegression(fit_intercept=False)
    regr.fit(X_max,y, sample_weight=weights)

    alphas = regr.coef_
    #quit()

    print('############')
    print(alphas)
    print('############')

    alphas[0] = 0


    # gust max from tuning (only time step with max zvp10 within an hour)
    gust_max_tune = zvp10_max + np.sum(alphas*X_max, axis=1)
    gust_max_tune = gust_max_tune.squeeze()
    gust_max_tune[gust_max_tune < 0] = 0

    if l_final_gust_timestepwise:
        # gust max from all time steps within an hour
        X_full = np.zeros( (X_max.shape[0], 360, X_max.shape[1]) )
        for i in range(0,360):
            X_full[:,i,:] = poly.fit_transform(X[:,i,:])[:,1:]
            if i_scaling:
                X_full[:,i,:] = scaler.transform(X_full[:,i,:])

        gust_max_fresh = np.max(zvp10 + np.sum(alphas*X_full, axis=2), axis=1)
        gust_max_fresh = gust_max_fresh.squeeze()
        gust_max_fresh[gust_max_fresh < 0] = 0




    # PLOT
    if i_plot > 0:
        #try:
        if l_final_gust_timestepwise:
            plot_error(obs_gust, model_mean, obs_mean, gust_max_fresh, gust_max_orig)
        else:
            plot_error(obs_gust, model_mean, obs_mean, gust_max_tune, gust_max_orig)
        plt.suptitle('READJUST  '+mode)

        if i_plot == 1:
            plt.show()
        elif i_plot > 1:
            if i_label == '':
                plot_name = CN.plot_path + 'tuning_readjNEW_'+str(mode)+'.png'
            else:
                plot_name = CN.plot_path + 'tuning_readjNEW_'+str(i_label)+'_'+str(mode)+'.png'
            print(plot_name)
            plt.savefig(plot_name)
            plt.close('all')
        #except:
        #    print('Tkinter ERROR while plotting!')


    # RESCALE ALPHA VALUES
    # not scaled

    quit()
    # SAVE PARAMETERS 
    if os.path.exists(CN.params_readjNEW_path):
        params = pickle.load( open(CN.params_readjNEW_path, 'rb') )
    else:
        params = {}
    params[mode] = {'alphas':{'1':alpha1,'2':alpha2}}
    pickle.dump(params, open(CN.params_readjNEW_path, 'wb'))


else:
    print('Train is turned off. Finish.')

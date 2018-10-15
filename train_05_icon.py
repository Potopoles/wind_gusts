import numpy as np
import os
import copy
import matplotlib.pyplot as plt
import pickle
from functions import plot_error, plot_mod_vs_obs
import globals as G
from namelist_cases import Case_Namelist
import namelist_cases as nl
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from functions_train import icon_feature_matrix
from datetime import timedelta

############ USER INPUT #############
case_index = nl.case_index
CN = Case_Namelist(case_index)
# do not plot (0) show plot (1) save plot (2)
i_plot = nl.i_plot
i_plot_type = nl.i_plot_type
model_dt = nl.model_dt
nhrs_forecast = nl.nhrs_forecast
i_scaling = 1
i_load = nl.i_load
i_train = nl.i_train
delete_existing_param_file = nl.delete_existing_param_file
#max_mean_wind_error = nl.max_mean_wind_error
#sample_weight = nl.sample_weight

modes = ['gust_mean',
        'gust_mean_mean2',
        'gust_mean_height',
        'gust_mean_mean2_height',
        'gust_mean_tkel1',
        'gust_mean_mean2_tkel1',
        'gust_mean_mean2_height_tkel1',
        'gust_mean_mean2_height_tkel1_dvl3v10',
        'gust_mean_mean2_height_dvl3v10',
        'gust_mean_mean2_tkel1_dvl3v10']

i_mode_ints = range(0,len(modes))
#i_mode_ints = [8]
#####################################

if delete_existing_param_file:
    try:
        os.remove(CN.params_icon_path)
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
    gust_ico = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    tkel1 = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    dvl3v10 = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    height = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    print('3D shape ' + str(gust_ico.shape))
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
                tkel1[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['tkel1'][ts_inds]
                dvl3v10[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['uvl3'][ts_inds] - \
                                              data[G.MODEL][G.STAT][stat_key][G.RAW] \
                                            [lm_run]['zvp10'][ts_inds]
                height[hr_ind,si,:] = data[G.STAT_META][stat_key]['hsurf'].values 

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
                #gust = zvp10 + ustar*ugn*( 1 + 0.5/12*hpbl*kappa*buoy/ustar**3 )**(1/3)

                gust_ico[hr_ind,si,:] = gust

            # OBSERVATION DATA
            full_hr_timestamps = data[G.MODEL][G.STAT][stat_keys[0]][G.RAW][lm_run]\
                                        ['tcm'].resample('H').max().index[1:]
            obs_gust[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_GUST_SPEED][full_hr_timestamps] 
            obs_mean[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_MEAN_WIND][full_hr_timestamps] 
    print()


    data = {}
    data['model_mean'] = model_mean
    data['gust_ico'] = gust_ico
    data['tkel1'] = tkel1
    data['dvl3v10'] = dvl3v10 
    data['height'] = height
    data['obs_gust'] = obs_gust
    data['obs_mean'] = obs_mean 

    pickle.dump(data, open(CN.train_icon_path, 'wb'))
else:
    data = pickle.load( open(CN.train_icon_path, 'rb') )

    model_mean = data['model_mean']
    model_mean = data['model_mean']
    gust_ico = data['gust_ico']
    tkel1 = data['tkel1']
    dvl3v10 = data['dvl3v10']
    height = data['height']
    obs_gust = data['obs_gust']
    obs_mean = data['obs_mean']



if i_train:


    obsmask = np.isnan(obs_gust)
    obsmask[np.isnan(obs_mean)] = True

    model_mean_hr = np.mean(model_mean, axis=2)

    ## bad mean wind accuracy mask
    #mean_abs_error = np.abs(model_mean_hr - obs_mean)
    #mean_rel_error = mean_abs_error/obs_mean
    #obsmask[mean_rel_error > max_mean_wind_error] = True

    obs_gust = obs_gust[~obsmask] 
    obs_mean = obs_mean[~obsmask] 
    model_mean = model_mean[~obsmask]
    model_mean_hr = model_mean_hr[~obsmask]
    gust_ico = gust_ico[~obsmask]
    tkel1 = tkel1[~obsmask]
    dvl3v10 = dvl3v10[~obsmask]
    height = height[~obsmask]
    N = obs_gust.flatten().shape[0]

    # find maximum gust
    maxid = gust_ico.argmax(axis=1)
    I = np.indices(maxid.shape)
    model_mean_max = model_mean[I,maxid].flatten() 
    gust_ico_max = gust_ico[I,maxid].flatten()
    gust_ico_max_unscaled = gust_ico[I,maxid].flatten()
    tkel1_max = tkel1[I,maxid].flatten()
    dvl3v10_max = dvl3v10[I,maxid].flatten()
    height_max = height[I,maxid].flatten()

    regr = LinearRegression(fit_intercept=False)

    for mode_int in i_mode_ints:
        mode = modes[mode_int]
        print('#################################################################################')
        print('############################## ' + str(mode) + ' ################################')

        # calc current time step gusts
        X = icon_feature_matrix(mode, gust_ico_max, height_max,
                                    dvl3v10_max, model_mean_max,
                                    tkel1_max)
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
            if i_plot_type == 0:
                plot_error(obs_gust, model_mean_hr, obs_mean, gust_max, gust_ico_max_unscaled)
            elif i_plot_type == 1:
                plot_mod_vs_obs(obs_gust, gust_max, gust_ico_max_unscaled)
            else:
                raise NotImplementedError()
            plt.suptitle('ICON  '+mode)

            if i_plot == 1:
                plt.show()
            elif i_plot > 1:
                if i_plot_type == 0:
                    plot_name = CN.plot_path + 'tuning_icon_'+str(mode)+'.png'
                elif i_plot_type == 1:
                    plot_name = CN.plot_path + 'plot1_tuning_icon_'+str(mode)+'.png'
                print(plot_name)
                plt.savefig(plot_name)
                plt.close('all')

        # RESCALE ALPHA VALUES
        # not necessary to treat powers > 1 different because
        # this is already contained in X matrix
        alphas = alphas/scaler.scale_
        print('alphas unscal  ' + str(alphas))

        # SAVE PARAMETERS 
        if os.path.exists(CN.params_icon_path):
            params = pickle.load( open(CN.params_icon_path, 'rb') )
        else:
            params = {}
        params[mode] = alphas
        pickle.dump(params, open(CN.params_icon_path, 'wb'))


else:
    print('Train is turned off. Finish.')

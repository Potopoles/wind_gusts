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
from functions_train import icon_feature_matrix

############ USER INPUT #############
case_index = 12
CN = Case_Namelist(case_index)
# do not plot (0) show plot (1) save plot (2)
i_plot = 2
model_dt = 10
i_scaling = 1
i_label = ''
i_load = 0
delete_existing_param_file = 1
modes = ['gust',
        'gust_gust2',
        'gust_gust2_height',
        'gust_gust2_tkel1',
        'gust_gust2_mean_tkel1',
        'gust_gust2_height_mean2',
        'gust_gust2_height_mean2_mean',
        'gust_gust2_height_mean2_mean_dvl3v10',
        'gust_gust2_height_mean2_mean_dvl3v10_tkel1',
        #'gust_mean2',
        'gust_height']
        #'gust_dvl3v10',
        #'gust_dvl3v10_height',
        #'gust_dvl3v10_height_mean2',
        #'gust_dvl3v10_height_mean2_gust2',

i_mode_ints = range(0,len(modes))
#i_mode_ints = [3,5]
#i_sample_weight = 'linear'
#i_sample_weight = 'squared'
i_sample_weight = '1'
max_mean_wind_error = 1.0
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
    n_hours = len(lm_runs)*24
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

    for lmi,lm_run in enumerate(lm_runs):
        print(lm_run)
        lm_inds = np.arange(lmi*24,(lmi+1)*24)
        model_hours_tmp = data[G.MODEL][G.STAT][stat_keys[0]][G.RAW][lm_run]\
                                    ['zvp10'].resample('H').max().index
        for si,stat_key in enumerate(stat_keys):
            # 3D
            tmp = data[G.MODEL][G.STAT][stat_key][G.RAW][lm_run]['zvp10']
            for hi,hour in enumerate(model_hours_tmp):
                loc_str = hour.strftime('%Y-%m-%d %H')
                hr_ind = lm_inds[hi]

                model_mean[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['zvp10'].loc[loc_str].values
                tkel1[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['tkel1'].loc[loc_str].values
                dvl3v10[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['uvl3'].loc[loc_str].values - \
                                              data[G.MODEL][G.STAT][stat_key][G.RAW] \
                                            [lm_run]['zvp10'].loc[loc_str].values
                height[hr_ind,si,:] = data[G.STAT_META][stat_key]['hsurf'].values 

                ugn = 7.71
                hpbl = 1000
                g = 9.80665
                Rd = 287.05
                etv = 0.6078
                cp = 1005.0
                kappa = Rd/cp

                umlev = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['ul1'].loc[loc_str].values 
                vmlev = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['vl1'].loc[loc_str].values 
                ps = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['ps'].loc[loc_str].values 
                qvflx = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['qvflx'].loc[loc_str].values 
                shflx = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['shflx'].loc[loc_str].values 
                z0 = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['z0'].loc[loc_str].values 
                Tskin = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['Tskin'].loc[loc_str].values 
                Tmlev = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['Tl1'].loc[loc_str].values 
                qvmlev = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['qvl1'].loc[loc_str].values 
                phimlev = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['phil1'].loc[loc_str].values 
                zvp10 = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['zvp10'].loc[loc_str].values 

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

            # 2D
            obs_gust[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_GUST_SPEED][model_hours_tmp] 
            obs_mean[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_MEAN_WIND][model_hours_tmp] 


    # observation to 1D and filter values
    obs_gust_flat = obs_gust.flatten()
    obs_mean_flat = obs_mean.flatten()
    obsmask = np.isnan(obs_gust_flat)
    obs_gust_flat = obs_gust_flat[~obsmask] 
    obs_mean_flat = obs_mean_flat[~obsmask] 
    N = obs_gust_flat.shape[0]

    # find maximum gust
    maxid = gust_ico.argmax(axis=2)
    I,J = np.indices(maxid.shape)

    gust_ico_max_unscaled = gust_ico[I,J,maxid].flatten()[~obsmask]

    model_mean_max = model_mean[I,J,maxid].flatten()[~obsmask] 
    model_mean = np.mean(model_mean, axis=2).flatten()[~obsmask]
    gust_ico_max = gust_ico[I,J,maxid].flatten()[~obsmask]
    tkel1_max = tkel1[I,J,maxid].flatten()[~obsmask]
    dvl3v10_max = dvl3v10[I,J,maxid].flatten()[~obsmask]
    height_max = height[I,J,maxid].flatten()[~obsmask]

    data = {}
    data['model_mean_max'] = model_mean_max
    data['model_mean'] = model_mean
    data['gust_ico_max'] = gust_ico_max
    data['tkel1_max'] = tkel1_max
    data['dvl3v10_max'] = dvl3v10_max 
    data['height_max'] = height_max
    data['obs_gust_flat'] = obs_gust_flat
    data['obs_mean_flat'] = obs_mean_flat 
    data['gust_ico_max_unscaled'] = gust_ico_max_unscaled

    pickle.dump(data, open(CN.train_icon_path, 'wb'))
else:
    data = pickle.load( open(CN.train_icon_path, 'rb') )

    model_mean_max = data['model_mean_max']
    model_mean = data['model_mean']
    gust_ico_max = data['gust_ico_max']
    tkel1_max = data['tkel1_max']
    dvl3v10_max = data['dvl3v10_max']
    height_max = data['height_max']
    obs_gust_flat = data['obs_gust_flat']
    obs_mean_flat = data['obs_mean_flat']
    gust_ico_max_unscaled = data['gust_ico_max_unscaled']

mean_abs_error = np.abs(model_mean - obs_mean_flat)
mean_rel_error = mean_abs_error/obs_mean_flat
errormask = mean_rel_error > max_mean_wind_error

model_mean_max = model_mean_max[~errormask]
model_mean = model_mean[~errormask]
gust_ico_max = gust_ico_max[~errormask]
tkel1_max = tkel1_max[~errormask]
dvl3v10_max = dvl3v10_max[~errormask]
height_max = height_max[~errormask]
obs_gust_flat = obs_gust_flat[~errormask]
obs_mean_flat = obs_mean_flat[~errormask]
gust_ico_max_unscaled = gust_ico_max_unscaled[~errormask]

regr = LinearRegression(fit_intercept=False)

for mode_int in i_mode_ints:
    mode = modes[mode_int]
    print('#################################################################################')
    print('############################## ' + str(mode) + ' ################################')

    # calc current time step gusts
    X = icon_feature_matrix(mode, gust_ico_max, height_max,
                                dvl3v10_max, model_mean_max,
                                tkel1_max)
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
        plot_error(obs_gust_flat, model_mean, obs_mean_flat, gust_max, gust_ico_max_unscaled)
        plt.suptitle('ICON  '+mode)

        if i_plot == 1:
            plt.show()
        elif i_plot > 1:
            if i_label == '':
                plot_name = CN.plot_path + 'tuning_icon_sw_'+i_sample_weight+'_mwa_'+str(max_mean_wind_error)+'_'\
                                            +str(mode)+'.png'
            else:
                plot_name = CN.plot_path + 'tuning_icon_sw_'+i_sample_weight+'_mwa_'+str(max_mean_wind_error)+'_'\
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
    if os.path.exists(CN.params_icon_path):
        params = pickle.load( open(CN.params_icon_path, 'rb') )
    else:
        params = {}
    params[mode] = alphas
    pickle.dump(params, open(CN.params_icon_path, 'wb'))

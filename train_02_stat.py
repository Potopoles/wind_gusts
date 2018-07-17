import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from functions import plot_error, scale
import globals as G
from namelist_cases import Case_Namelist

############ USER INPUT #############
case_index = 6
CN = Case_Namelist(case_index)
#min_gust = 10
# do not plot (0) show plot (1) save plot (2)
i_plot = 2
model_dt = 10
i_label = ''
i_load = 1
i_output_error = 1
learning_rate_factor = 1E-3
d_error_thresh = 1E-5
i_sample_weight = '1'

modes = ['mean',
         'mean_mean2',
         #'mean_tcm',
         #'mean_height',
         #'mean_sso',
         'mean_tkebra',
         'mean_tkebra_mean2',
         'mean_gustbra',
         'mean_gustbra_dvl3v10',
         'mean_kbra',
         'mean_dvl3v10',
         'mean_icon',
         'mean_gustbra_icon',
         #'mean_z0',
         'mean_gustbra_mean2',
         #'mean_gustbra_mean2_icon',
         'mean_gustbra_mean2_kbra',
         'mean_gustbra_mean2_tkebra',
         'mean_gustbra_mean2_tkebrakbra']
i_mode_ints = range(0,len(modes))
#i_mode_ints = [len(modes)-1]
#i_mode_ints = [12]
max_mean_wind_error = 1.0
#max_mean_wind_error = 0.3
sgd_prob = 0.02
feature_names = ['tcm','zvp10', 'hsurf', 'sso_stdh', 'tke_bra_es', 'zv_bra_es', 'k_bra_es', 'dvl3v10', 'z0', \
                'icon_gust']

def calculate_gust(mode, features, alphas, zvp10_unsc):
    if mode == 'mean':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10']
    elif mode == 'mean_mean2':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['zvp10']**2
    #elif mode == 'mean_tcm':
    #    gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['tcm']
    #elif mode == 'mean_height':
    #    gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['hsurf']
    #elif mode == 'mean_sso':
    #    gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['sso_stdh']
    elif mode == 'mean_tkebra':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['tke_bra_es']
    elif mode == 'mean_tkebra_mean2':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['tke_bra_es'] \
                + alphas[3]*features['zvp10']**2
    elif mode == 'mean_gustbra':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['zv_bra_es']
    elif mode == 'mean_gustbra_dvl3v10':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['zv_bra_es'] \
                + alphas[3]*features['dvl3v10']
    elif mode == 'mean_kbra':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['k_bra_es']
    elif mode == 'mean_dvl3v10':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['dvl3v10']
    elif mode == 'mean_icon':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['icon_gust']
    elif mode == 'mean_gustbra_icon':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['zv_bra_es'] + \
                + alphas[3]*features['icon_gust']
    #elif mode == 'mean_z0':
    #    gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['z0']
    elif mode == 'mean_gustbra_mean2':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['zv_bra_es'] \
                + alphas[3]*features['zvp10']**2
    elif mode == 'mean_gustbra_mean2_icon':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['zv_bra_es'] \
                + alphas[3]*features['zvp10']**2 + alphas[4]*features['icon_gust']
    elif mode == 'mean_gustbra_mean2_kbra':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['zv_bra_es'] \
                + alphas[3]*features['zvp10']**2 + alphas[4]*features['k_bra_es']
    elif mode == 'mean_gustbra_mean2_tkebra':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['zv_bra_es'] \
                + alphas[3]*features['zvp10']**2 + alphas[4]*features['tke_bra_es']
    elif mode == 'mean_gustbra_mean2_tkebrakbra':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['zv_bra_es'] \
                + alphas[3]*features['zvp10']**2 + alphas[4]*features['tke_bra_es']*features['k_bra_es']
    else:
        raise ValueError('wrong mode')
    return(gust)

def combine_features(mode, features, zvp10_unsc):
    if mode == 'mean':
        X = np.full((zvp10_unsc.shape[0],2), np.nan)
        X[:,1] = features['zvp10']
    elif mode == 'mean_mean2':
        X = np.full((zvp10_unsc.shape[0],3), np.nan)
        X[:,1] = features['zvp10']
        X[:,2] = features['zvp10']**2
    #elif mode == 'mean_tcm':
    #    X = np.full((zvp10_unsc.shape[0],3), np.nan)
    #    X[:,1] = features['zvp10']
    #    X[:,2] = features['tcm']
    #elif mode == 'mean_height':
    #    X = np.full((zvp10_unsc.shape[0],3), np.nan)
    #    X[:,1] = features['zvp10']
    #    X[:,2] = features['hsurf']
    #elif mode == 'mean_sso':
    #    X = np.full((zvp10_unsc.shape[0],3), np.nan)
    #    X[:,1] = features['zvp10']
    #    X[:,2] = features['sso_stdh']
    elif mode == 'mean_tkebra':
        X = np.full((zvp10_unsc.shape[0],3), np.nan)
        X[:,1] = features['zvp10']
        X[:,2] = features['tke_bra_es']
    elif mode == 'mean_tkebra_mean2':
        X = np.full((zvp10_unsc.shape[0],4), np.nan)
        X[:,1] = features['zvp10']
        X[:,2] = features['tke_bra_es']
        X[:,3] = features['zvp10']**2
    elif mode == 'mean_gustbra':
        X = np.full((zvp10_unsc.shape[0],3), np.nan)
        X[:,1] = features['zvp10']
        X[:,2] = features['zv_bra_es']
    elif mode == 'mean_gustbra_dvl3v10':
        X = np.full((zvp10_unsc.shape[0],4), np.nan)
        X[:,1] = features['zvp10']
        X[:,2] = features['zv_bra_es']
        X[:,3] = features['dvl3v10']
    elif mode == 'mean_kbra':
        X = np.full((zvp10_unsc.shape[0],3), np.nan)
        X[:,1] = features['zvp10']
        X[:,2] = features['k_bra_es']
    elif mode == 'mean_dvl3v10':
        X = np.full((zvp10_unsc.shape[0],3), np.nan)
        X[:,1] = features['zvp10']
        X[:,2] = features['dvl3v10']
    elif mode == 'mean_icon':
        X = np.full((zvp10_unsc.shape[0],3), np.nan)
        X[:,1] = features['zvp10']
        X[:,2] = features['icon_gust']
    elif mode == 'mean_gustbra_icon':
        X = np.full((zvp10_unsc.shape[0],4), np.nan)
        X[:,1] = features['zvp10']
        X[:,2] = features['zv_bra_es']
        X[:,3] = features['icon_gust']
    #elif mode == 'mean_z0':
    #    X = np.full((zvp10_unsc.shape[0],3), np.nan)
    #    X[:,1] = features['zvp10']
    #    X[:,2] = features['z0']
    elif mode == 'mean_gustbra_mean2':
        X = np.full((zvp10_unsc.shape[0],4), np.nan)
        X[:,1] = features['zvp10']
        X[:,2] = features['zv_bra_es']
        X[:,3] = features['zvp10']**2
    elif mode == 'mean_gustbra_mean2_icon':
        X = np.full((zvp10_unsc.shape[0],5), np.nan)
        X[:,1] = features['zvp10']
        X[:,2] = features['zv_bra_es']
        X[:,3] = features['zvp10']**2
        X[:,4] = features['icon_gust']
    elif mode == 'mean_gustbra_mean2_kbra':
        X = np.full((zvp10_unsc.shape[0],5), np.nan)
        X[:,1] = features['zvp10']
        X[:,2] = features['zv_bra_es']
        X[:,3] = features['zvp10']**2
        X[:,4] = features['k_bra_es']
    elif mode == 'mean_gustbra_mean2_tkebra':
        X = np.full((zvp10_unsc.shape[0],5), np.nan)
        X[:,1] = features['zvp10']
        X[:,2] = features['zv_bra_es']
        X[:,3] = features['zvp10']**2
        X[:,4] = features['tke_bra_es']
    elif mode == 'mean_gustbra_mean2_tkebrakbra':
        X = np.full((zvp10_unsc.shape[0],5), np.nan)
        X[:,1] = features['zvp10']
        X[:,2] = features['zv_bra_es']
        X[:,3] = features['zvp10']**2
        X[:,4] = features['tke_bra_es']*features['k_bra_es']
    X[:,0] = 1
    return(X)
#####################################

# create directories
if i_plot > 1 and not os.path.exists(CN.plot_path):
    os.mkdir(CN.plot_path)


if not i_load:
    # load data
    data = pickle.load( open(CN.mod_path, 'rb') )

    #stat_keys = ['AEG','ABO', 'AIG']
    #stat_keys = ['ABO', 'AEG']
    stat_keys = data[G.STAT_NAMES]

    lm_runs = list(data[G.MODEL][G.STAT][stat_keys[0]][G.RAW].keys())
    n_hours = len(lm_runs)*24
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

    for lmi,lm_run in enumerate(lm_runs):
        print(lm_run)
        lm_inds = np.arange(lmi*24,(lmi+1)*24)
        model_hours_tmp = data[G.MODEL][G.STAT][stat_keys[0]][G.RAW][lm_run]\
                                    ['tcm'].resample('H').max().index
        for si,stat_key in enumerate(stat_keys):
            # 3D
            tmp = data[G.MODEL][G.STAT][stat_key][G.RAW][lm_run]['tcm']
            for hi,hour in enumerate(model_hours_tmp):
                loc_str = hour.strftime('%Y-%m-%d %H')
                hr_ind = lm_inds[hi]

                for feat in feature_names:
                    if feat in ['zvp10', 'tcm', 'tke_bra_es', 'zv_bra_es', 'k_bra_es']:
                        features[feat][hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                                    [lm_run][feat].loc[loc_str].values
                    elif feat in ['hsurf', 'sso_stdh', 'z0']:
                        features[feat][hr_ind,si,:] = data[G.STAT_META][stat_key][feat].values 
                    elif feat == 'dvl3v10':
                        features[feat][hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                                    [lm_run]['uvl3'].loc[loc_str].values - \
                                                      data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                                    [lm_run]['zvp10'].loc[loc_str].values
                    elif feat == 'icon_gust':
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
                        features[feat][hr_ind,si,:] = gust

            # 2D
            obs_gust[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_GUST_SPEED][model_hours_tmp] 
            obs_mean[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_MEAN_WIND][model_hours_tmp] 


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


# obs nan mask
obsmask = np.isnan(obs_gust)
obsmask[np.isnan(obs_mean)] = True

# bad mean wind accuracy mask
mean_abs_error = np.abs(model_mean - obs_mean)
mean_rel_error = mean_abs_error/obs_mean
errormask = mean_rel_error > max_mean_wind_error
# combine both
obsmask[errormask] = True

obs_gust = obs_gust[~obsmask] 
obs_mean = obs_mean[~obsmask]
model_mean = model_mean[~obsmask]
for feat in feature_names:
    features[feat] = features[feat][~obsmask]

# initial gust
gust = features['zvp10'] + 7.2*features['tcm']*features['zvp10']
gust_max_unscaled = np.amax(gust,axis=1)


# SCALING
zvp10_unsc = copy.deepcopy(features['zvp10'])
for feat in feature_names:
    features[feat] = scale(features[feat])


for mode_int in i_mode_ints:
    mode = modes[mode_int]
    print('#################################################################################')
    print('############################## ' + str(mode) + ' ################################')

    alphas = {0:0,1:0,2:0,3:0,4:0}
    error_old = np.Inf
    #d_error = 100
    d_errors = np.full(int(1/sgd_prob*6), 100.)
    learning_rate = 1E-2
    #d_error_thresh = learning_rate_factor

    #alpha0 = []
    #alpha1 = []
    #alpha2 = []

    c = 0
    while np.abs(np.mean(d_errors)) > d_error_thresh:

        # SGD selection
        sgd_inds = np.random.choice([True, False], (features['zvp10'].shape[0]), p=[sgd_prob,1-sgd_prob])

        sgd_zvp10_unsc = zvp10_unsc[sgd_inds,:]
        sgd_obs_gust = obs_gust[sgd_inds]

        N = len(sgd_obs_gust)

        sgd_features = {}
        for feat in feature_names:
            sgd_features[feat] = features[feat][sgd_inds,:]

        # calc current time step gusts
        sgd_gust = calculate_gust(mode, sgd_features, alphas, sgd_zvp10_unsc)

        # find maximum gust
        maxid = sgd_gust.argmax(axis=1)
        I = np.indices(maxid.shape)
        sgd_gust_max = sgd_gust[I,maxid].squeeze()

        for feat in feature_names:
            sgd_features[feat] = sgd_features[feat][I,maxid].squeeze()

        X = combine_features(mode, sgd_features, sgd_zvp10_unsc)

        # error
        deviation = sgd_obs_gust - sgd_gust_max
        error_now = np.sqrt(np.sum(deviation**2)/N)
        d_error = error_old - error_now
        d_errors = np.roll(d_errors, shift=1)
        d_errors[0] = d_error
        error_old = error_now
        if i_output_error:
            if c % 10 == 0:
                print(str(c) + '\t' + str(error_now) + '\t' + str(np.abs(np.mean(d_errors))))

        if i_sample_weight == 'linear':
            weights = sgd_obs_gust
        elif i_sample_weight == 'squared':
            weights = sgd_obs_gust**2
        else:
            weights = np.repeat(1,len(sgd_obs_gust))

        for i in range(0,X.shape[1]):
            #dalpha = -2/N * np.sum( X[:,i] * deviation )
            dalpha = -2/np.sum(weights) * np.sum( X[:,i] * deviation * weights )
            alphas[i] = alphas[i] - learning_rate * dalpha
        #alpha0.append(alphas[0])
        #alpha1.append(alphas[1])
        #alpha2.append(alphas[2])

        # adjust learning rate
        learning_rate = error_now*learning_rate_factor

        c += 1

    #plt.plot(alpha0)
    #plt.plot(alpha1)
    #plt.plot(alpha2)
    #plt.show()
    print('############')
    print(alphas)
    print('############')

    # Calculate final gust
    gust = calculate_gust(mode, features, alphas, zvp10_unsc)
    maxid = gust.argmax(axis=1)
    I = np.indices(maxid.shape)
    gust_max = gust[I,maxid].squeeze()
    
    plot_error(obs_gust, model_mean, obs_mean, gust_max, gust_max_unscaled)
    plt.suptitle('STAT  '+mode)

    if i_plot == 1:
        plt.show()
    elif i_plot > 1:
        if i_label == '':
            plot_name = CN.plot_path + 'tuning_stat_sw_'+i_sample_weight+'_'+str(mode)+'.png'
        else:
            plot_name = CN.plot_path + 'tuning_stat_sw_'+i_sample_weight+'_'+str(i_label)+'_'+str(mode)+'.png'
        print(plot_name)
        plt.savefig(plot_name)
        plt.close('all')


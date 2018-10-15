import numpy as np
import os
import copy
import matplotlib.pyplot as plt
import pickle
from functions import plot_error, plot_mod_vs_obs
import globals as G
from namelist_cases import Case_Namelist
import namelist_cases as nl
from datetime import timedelta

############ USER INPUT #############
case_index = nl.case_index
CN = Case_Namelist(case_index)
# do not plot (0) show plot (1) save plot (2)
i_plot = nl.i_plot
i_plot_type = nl.i_plot_type
model_dt = nl.model_dt
nhrs_forecast = nl.nhrs_forecast
i_load = nl.i_load
i_train = nl.i_train
delete_existing_param_file = nl.delete_existing_param_file
#max_mean_wind_error = nl.max_mean_wind_error
#sample_weight = nl.sample_weight

#modes = ['es',
#        'lb',
#        'es_rho',
#        'lb_rho']
modes = ['es',
        'lb']

i_mode_ints = range(0,len(modes))
#i_mode_ints = [2]
#####################################

# create directories
if i_plot > 1 and not os.path.exists(CN.plot_path):
    os.mkdir(CN.plot_path)

if delete_existing_param_file:
    try:
        os.remove(CN.params_phys_bra_path)
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
    kval_est = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    gust_est = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    kval_lb = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    gust_lb = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    #rho_est = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    #rho_lb = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    #rho_surf = np.full((n_hours, n_stats, ts_per_hour), np.nan)
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
                kval_lb[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['k_bra_lb'][ts_inds]
                gust_est[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['zv_bra_es'][ts_inds]
                gust_lb[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['zv_bra_lb'][ts_inds]
                #rho_est[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                #                            [lm_run]['rho_bra_es'][ts_inds]
                #rho_lb[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                #                            [lm_run]['rho_bra_lb'][ts_inds]
                #rho_surf[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                #                            [lm_run]['rhol1'][ts_inds]

            # OBSERVATION DATA
            full_hr_timestamps = data[G.MODEL][G.STAT][stat_keys[0]][G.RAW][lm_run]\
                                        ['tcm'].resample('H').max().index[1:]
            obs_gust[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_GUST_SPEED][full_hr_timestamps] 
            obs_mean[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_MEAN_WIND][full_hr_timestamps] 
    print()


    # Process fields
    kheight_est = copy.deepcopy(kval_est)
    kheight_lb = copy.deepcopy(kval_lb)
    kalts = np.loadtxt('../data/kaltitudes.txt')
    kinds = kalts[:,0].astype(np.int)
    kalts = kalts[:,1]
    for i,kind in enumerate(kinds):
        kheight_est[kval_est == kind] = kalts[i]
        kheight_lb[kval_lb == kind] = kalts[i]

    model_mean_hr = np.mean(model_mean, axis=2)

    data = {}
    data['model_mean_hr'] = model_mean_hr
    data['model_mean'] = model_mean
    data['gust_est'] = gust_est
    data['gust_lb'] = gust_lb
    data['kheight_est'] = kheight_est
    data['kheight_lb'] = kheight_lb
    #data['rho_est'] = rho_est
    #data['rho_lb'] = rho_lb
    #data['rho_surf'] = rho_surf
    data['obs_gust'] = obs_gust
    data['obs_mean'] = obs_mean 

    pickle.dump(data, open(CN.phys_bra_path, 'wb'))

else:
    data = pickle.load( open(CN.phys_bra_path, 'rb') )

    model_mean_hr = data['model_mean_hr']
    model_mean = data['model_mean']
    gust_est = data['gust_est']
    gust_lb = data['gust_lb']
    kheight_est = data['kheight_est']
    kheight_lb = data['kheight_lb']
    #rho_est = data['rho_est']
    #rho_lb = data['rho_lb']
    #rho_surf = data['rho_surf']
    obs_gust = data['obs_gust']
    obs_mean = data['obs_mean']




if i_train:

    # obs nan mask
    obsmask = np.isnan(obs_gust)
    obsmask[np.isnan(obs_mean)] = True

    ## bad mean wind accuracy mask
    #mean_abs_error = np.abs(model_mean_hr - obs_mean)
    #mean_rel_error = mean_abs_error/obs_mean
    #errormask = mean_rel_error > max_mean_wind_error
    ## combine both
    #obsmask[errormask] = True

    obs_gust = obs_gust[~obsmask] 
    obs_mean = obs_mean[~obsmask]
    model_mean_hr = model_mean_hr[~obsmask]
    model_mean = model_mean[~obsmask]
    kheight_est = kheight_est[~obsmask]
    gust_est = gust_est[~obsmask]
    kheight_lb = kheight_lb[~obsmask]
    gust_lb = gust_lb[~obsmask]
    #rho_est = rho_est[~obsmask]
    #rho_lb = rho_lb[~obsmask]
    #rho_surf = rho_surf[~obsmask]

    for mode_int in i_mode_ints:
        mode = modes[mode_int]
        print('#################################################################################')
        print('############################## ' + str(mode) + ' ################################')

        sgd_prob = 0.05
        learning_rate_factor = 1E-10
        d_error_thresh = 1E-5
        i_output_error = 1
        learning_rate = 1E-10

        alphas = {0:0}
        error_old = np.Inf
        d_errors = np.full(int(1/sgd_prob*6), 100.)

        c = 0
        while np.abs(np.mean(d_errors)) > d_error_thresh:

            # SGD selection
            sgd_inds = np.random.choice([True, False], (gust_est.shape[0]), p=[sgd_prob,1-sgd_prob])

            #sgd_zvp10_unsc = zvp10_unsc[sgd_inds,:]
            sgd_obs_gust = obs_gust[sgd_inds]

            N = len(sgd_obs_gust)

            sgd_model_mean = model_mean[sgd_inds,:]
            sgd_gust_est = gust_est[sgd_inds,:]
            sgd_kheight_est = kheight_est[sgd_inds,:]
            #sgd_rho_est = rho_est[sgd_inds,:]
            #sgd_rho_lb = rho_lb[sgd_inds,:]
            #sgd_rho_surf = rho_surf[sgd_inds,:]
            sgd_gust_lb = gust_lb[sgd_inds,:]
            sgd_kheight_lb = kheight_lb[sgd_inds,:]

            # calc current time step gusts
            if mode == 'es':
                sgd_gust = sgd_gust_est - alphas[0]*sgd_kheight_est*(sgd_gust_est -  sgd_model_mean)
            if mode == 'lb':
                sgd_gust = sgd_gust_lb - alphas[0]*sgd_kheight_lb*(sgd_gust_lb -  sgd_model_mean)
            #elif mode == 'es_rho':
            #    sgd_gust = sgd_gust_est*sgd_rho_est/sgd_rho_surf - alphas[0]*sgd_kheight_est*\
            #                (sgd_gust_est*sgd_rho_est/sgd_rho_surf -  sgd_model_mean)
            #elif mode == 'lb_rho':
            #    sgd_gust = sgd_gust_lb*sgd_rho_lb/sgd_rho_surf - alphas[0]*sgd_kheight_lb*\
            #                (sgd_gust_lb*sgd_rho_lb/sgd_rho_surf -  sgd_model_mean)

            sgd_gust[sgd_gust < 0] = 0

            # find maximum gust
            maxid = sgd_gust.argmax(axis=1)
            I = np.indices(maxid.shape)
            sgd_gust_max = sgd_gust[I,maxid].squeeze()

            sgd_model_mean = model_mean[sgd_inds,:][I,maxid].squeeze()
            sgd_gust_est = gust_est[sgd_inds,:][I,maxid].squeeze()
            sgd_kheight_est = kheight_est[sgd_inds,:][I,maxid].squeeze()
            #sgd_rho_est = rho_est[sgd_inds,:][I,maxid].squeeze()
            #sgd_rho_lb = rho_lb[sgd_inds,:][I,maxid].squeeze()
            #sgd_rho_surf = rho_surf[sgd_inds,:][I,maxid].squeeze()
            sgd_kheight_lb = kheight_lb[sgd_inds,:][I,maxid].squeeze()
            sgd_gust_lb = gust_lb[sgd_inds,:][I,maxid].squeeze()

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

            if mode == 'es':
                dalpha = -2/N * np.sum( -sgd_kheight_est*(sgd_gust_est -  sgd_model_mean) * deviation )
            if mode == 'lb':
                dalpha = -2/N * np.sum( -sgd_kheight_lb*(sgd_gust_lb -  sgd_model_mean) * deviation )
            #elif mode == 'es_rho':
            #    dalpha = -2/N * np.sum( -sgd_kheight_est*\
            #                            (sgd_gust_est*sgd_rho_est/sgd_rho_surf -  sgd_model_mean) * deviation )
            #elif mode == 'lb_rho':
            #    dalpha = -2/N * np.sum( -sgd_kheight_lb*\
            #                            (sgd_gust_lb*sgd_rho_lb/sgd_rho_surf -  sgd_model_mean) * deviation )

            alphas[0] = alphas[0] - learning_rate * dalpha

            # adjust learning rate
            learning_rate = error_now*learning_rate_factor

            c += 1

        print('alpha ' + str(alphas))


        # final gust
        if mode == 'es':
            gust = gust_est - alphas[0]*kheight_est*(gust_est - model_mean)
        if mode == 'lb':
            gust = gust_lb - alphas[0]*kheight_lb*(gust_lb - model_mean)
        #elif mode == 'es_rho':
        #    gust = gust_est*rho_est/rho_surf - alphas[0]*kheight_est*\
        #                                (gust_est*rho_est/rho_surf -  model_mean)
        #elif mode == 'lb_rho':
        #    gust = gust_lb*rho_lb/rho_surf - alphas[0]*kheight_lb*\
        #                                (gust_lb*rho_lb/rho_surf -  model_mean)
            
        gust[gust < 0] = 0

        # find maximum gust
        maxid = gust.argmax(axis=1)
        I = np.indices(maxid.shape)
        gust_max = gust[I,maxid].squeeze()

        #if mode in ['lb', 'lb_rho']:
        if mode in ['lb']:
            maxid = gust_lb.argmax(axis=1)
            I = np.indices(maxid.shape)
            gust_max_orig = gust_lb[I,maxid].squeeze()
            suptitle = 'PHY BRALB  '
            plot_name_title = 'tuning_phys_bralb_'
        else:
            maxid = gust_est.argmax(axis=1)
            I = np.indices(maxid.shape)
            gust_max_orig = gust_est[I,maxid].squeeze()
            suptitle = 'PHY BRAES  '
            plot_name_title = 'tuning_phys_braes_'

        if i_plot > 0:
            if i_plot_type == 0:
                plot_error(obs_gust, model_mean_hr, obs_mean, gust_max, gust_max_orig)
            elif i_plot_type == 1:
                plot_mod_vs_obs(obs_gust, gust_max, gust_max_orig)
            else:
                raise NotImplementedError()
            plt.suptitle(suptitle + mode)

            if i_plot == 1:
                plt.show()
            elif i_plot > 1:
                if i_plot_type == 0:
                    plot_name = CN.plot_path + plot_name_title +str(mode)+'.png'
                elif i_plot_type == 1:
                    plot_name = CN.plot_path + 'plot1_'+plot_name_title+str(mode)+'.png'

                print(plot_name)
                plt.savefig(plot_name)
                plt.close('all')

        # SAVE PARAMETERS 
        if os.path.exists(CN.params_phys_bra_path):
            params = pickle.load( open(CN.params_phys_bra_path, 'rb') )
        else:
            params = {}
        params[mode] = alphas
        pickle.dump(params, open(CN.params_phys_bra_path, 'wb'))


else:
    print('Train is turned off. Finish.')






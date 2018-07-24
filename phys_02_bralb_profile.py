import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import pickle
from functions import plot_error
import globals as G
from namelist_cases import Case_Namelist

############ USER INPUT #############
case_index = 0
CN = Case_Namelist(case_index)
#min_gust = 10
# do not plot (0) show plot (1) save plot (2)
i_plot = 2
model_dt = 10
#i_scaling = 1
i_label = ''
i_load = 1

max_mean_wind_error = 1.0
#max_mean_wind_error = 0.5
#max_mean_wind_error = 0.3

alpha = 0.00001
d_alpha = 0.00001

alpha = 0.0001
d_alpha = 0.0001

alpha = 0

bra_modes = ['lb', 'es']
tke_modes = ['', 'tke']
rho_modes = ['', 'rho']

modes = []
for bra_mode in bra_modes:
    for tke_mode in tke_modes:
        for rho_mode in rho_modes:
            modes.append({'bra':bra_mode,
                        'tke':tke_mode,
                        'rho':rho_mode})

#modes = ['lb_no_tke',
#        'lb_tke',
#        'lb_no_tke_rho',
#        'lb_tke_rho',
#        'es_no_tke',
#        'es_tke',
#        'es_no_tke_rho',
#        'es_tke_rho']
i_mode_ints = range(0,len(modes))
#####################################


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
    n_vertical = 40


    # 3D
    model_mean = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    kval_lb = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    gust_lb = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    kval_es = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    gust_es = np.full((n_hours, n_stats, ts_per_hour), np.nan)
    print('3D shape ' + str(kval_lb.shape))
    # 4D
    wind_k = np.full((n_hours, n_stats, ts_per_hour, n_vertical), np.nan)
    tke_k = np.full((n_hours, n_stats, ts_per_hour, n_vertical), np.nan)
    rho_k = np.full((n_hours, n_stats, ts_per_hour, n_vertical), np.nan)
    print('4D shape ' + str(wind_k.shape))
    # 2D
    obs_gust = np.full((n_hours, n_stats), np.nan)
    obs_mean = np.full((n_hours, n_stats), np.nan)

    # Retrieve altitude thickness of model levels
    kalts = np.loadtxt('../data/kaltitudes.txt')
    k_inds_file = kalts[:,0].astype(np.int)
    k_alts_file = kalts[:,1]
    k_inds = np.arange(80,80-n_vertical,-1)
    k_dz = np.full( n_vertical, np.nan)
    for ki,k_ind in enumerate(k_inds):
        k_dz[ki] = k_alts_file[k_inds_file == k_ind] - k_alts_file[k_inds_file == k_ind+1]
    #print(k_dz)
    #print(k_inds)

    # GET DATA
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

                # 3D
                model_mean[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['zvp10'].loc[loc_str].values
                kval_lb[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['k_bra_lb'].loc[loc_str].values
                gust_lb[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['zv_bra_lb'].loc[loc_str].values
                kval_es[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['k_bra_es'].loc[loc_str].values
                gust_es[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                                            [lm_run]['zv_bra_es'].loc[loc_str].values

                # 4D
                #wind_k[hr_ind,si,:] = data[G.MODEL][G.STAT][stat_key][G.RAW]\
                #                            [lm_run]['k_bra_lb'].loc[loc_str].values
                hr_inds_4d = np.arange(hi*3600/model_dt,(hi+1)*3600/model_dt).astype(np.int)
                #print(hr_inds_4d)
                #print(data[G.MODEL][G.STAT][stat_key][G.RAW2][lm_run].shape)
                #print(data[G.MODEL][G.STAT][stat_key][G.RAW2][lm_run][hr_inds_4d,0:30].shape)
                wind_k[hr_ind,si,:,:] = data[G.MODEL][G.STAT][stat_key][G.RAW2][lm_run][hr_inds_4d,0:40]
                tke_k[hr_ind,si,:,:] = data[G.MODEL][G.STAT][stat_key][G.RAW2][lm_run][hr_inds_4d,40:80]
                rho_k[hr_ind,si,:,:] = data[G.MODEL][G.STAT][stat_key][G.RAW2][lm_run][hr_inds_4d,80:120]
                #print(wind_k)
                #quit()

            # 2D
            obs_gust[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_GUST_SPEED][model_hours_tmp] 
            obs_mean[lm_inds,si] = data[G.OBS][G.STAT][stat_key][G.OBS_MEAN_WIND][model_hours_tmp] 


    # obs nan mask
    obsmask = np.isnan(obs_gust)
    obsmask[np.isnan(obs_mean)] = True

    # bad mean wind accuracy mask
    model_mean_hr = np.mean(model_mean, axis=2)
    mean_abs_error = np.abs(model_mean_hr - obs_mean)
    mean_rel_error = mean_abs_error/obs_mean
    errormask = mean_rel_error > max_mean_wind_error
    # combine both
    obsmask[errormask] = True

    # 2D
    obs_gust = obs_gust[~obsmask] 
    obs_mean = obs_mean[~obsmask]
    model_mean_hr = model_mean_hr[~obsmask]
    # 3D
    model_mean = model_mean[~obsmask]
    kval_lb = kval_lb[~obsmask]
    gust_lb = gust_lb[~obsmask]
    kval_es = kval_es[~obsmask]
    gust_es = gust_es[~obsmask]
    # 4D
    wind_k = wind_k[~obsmask]
    tke_k = tke_k[~obsmask]
    rho_k = rho_k[~obsmask]


    tke_k[tke_k < 0.1] = 0.1


    data = {}
    data['model_mean_hr'] = model_mean_hr
    data['model_mean'] = model_mean
    data['gust_lb'] = gust_lb
    data['kval_lb'] = kval_lb
    data['gust_es'] = gust_es
    data['kval_es'] = kval_es
    data['obs_gust'] = obs_gust
    data['obs_mean'] = obs_mean 
    data['wind_k'] = wind_k
    data['tke_k'] = tke_k
    data['rho_k'] = rho_k

    data['k_inds'] = k_inds
    data['k_dz'] = k_dz

    pickle.dump(data, open(CN.phys_braprof_path, 'wb'))

else:
    data = pickle.load( open(CN.phys_braprof_path, 'rb') )

    model_mean_hr = data['model_mean_hr']
    model_mean = data['model_mean']
    gust_lb = data['gust_lb']
    kval_lb = data['kval_lb']
    gust_es = data['gust_es']
    kval_es = data['kval_es']
    obs_gust = data['obs_gust']
    obs_mean = data['obs_mean']
    wind_k = data['wind_k']
    tke_k = data['tke_k']
    rho_k = data['rho_k']

    k_inds = data['k_inds']
    k_dz = data['k_dz']



N = obs_gust.shape[0]


def reduce_wind(kval, gust, wind_col, tke_col, wind_l0, k_inds, k_dz, alpha, mode): 
    between_inds = np.arange(1,np.argwhere(k_inds == kval).squeeze())
    winds_between = wind_col[between_inds]
    if mode in ['lb_tke', 'es_tke']:
        tke_between = tke_col[between_inds]
    wind_diff = gust - winds_between
    if mode in ['lb_tke', 'es_tke']:
        reduction_between = - alpha * tke_between * wind_diff * k_dz[between_inds]
    else:
        reduction_between = - alpha * wind_diff * k_dz[between_inds]
    reduction = np.sum(reduction_between)

    final_gust = gust + reduction
    if final_gust < wind_l0:
        final_gust = wind_l0
    #if np.isnan(final_gust):
    #    quit()
    return(final_gust)


def reduce_wind_rho(kval, gust, wind_col, tke_col, rho_col, wind_l0, k_inds, k_dz, alpha, mode): 
    source_ind = np.argwhere(k_inds == kval).squeeze()
    between_inds = np.arange(1,source_ind)
    winds_between = wind_col[between_inds]
    rho_between = rho_col[between_inds]
    rho = rho_col[source_ind]
    if mode in ['lb_tke_rho', 'es_tke_rho']:
        tke_between = tke_col[between_inds]
    wind_diff = gust - winds_between
    if mode in ['lb_tke_rho', 'es_tke_rho']:
        reduction_between = - alpha * tke_between * rho_between/rho * wind_diff * k_dz[between_inds]
    else:
        reduction_between = - alpha * rho_between/rho * wind_diff * k_dz[between_inds]
    reduction = np.sum(reduction_between)

    final_gust = gust + reduction
    if final_gust < wind_l0:
        final_gust = wind_l0
    #if np.isnan(final_gust):
    #    quit()
    return(final_gust)


alphas = []
errors = []


for mode_int in i_mode_ints:
    mode_entry = modes[mode_int]
    mode = mode_entry['bra'] + '_' + mode_entry['tke'] + '_' + mode_entry['rho']
    print('#################################################################################')
    print('############################## ' + str(mode) + ' ################################')


    error_old = 1000
    error_now = 100



    c = 0
    while error_now < error_old:
        c += 1

        gust_ts = np.full( model_mean.shape, np.nan)
        gust_hr = np.full( model_mean_hr.shape, np.nan)

        for hr_ind in range(0,gust_ts.shape[0]):
            for ts_ind in range(0,gust_ts.shape[1]):

                if mode in ['lb_tke_', 'lb__']:
                    gust_ts[hr_ind, ts_ind] = \
                                reduce_wind(kval_lb[hr_ind,ts_ind], gust_lb[hr_ind,ts_ind], \
                                wind_k[hr_ind,ts_ind,:], tke_k[hr_ind,ts_ind,:], \
                                model_mean[hr_ind,ts_ind], k_inds, k_dz, alpha, mode)
                elif mode in ['lb_tke_rho', 'lb__rho']:
                    gust_ts[hr_ind, ts_ind] = \
                                reduce_wind_rho(kval_lb[hr_ind,ts_ind], gust_lb[hr_ind,ts_ind], \
                                wind_k[hr_ind,ts_ind,:], tke_k[hr_ind,ts_ind,:], rho_k[hr_ind,ts_ind,:], \
                                model_mean[hr_ind,ts_ind], k_inds, k_dz, alpha, mode)
                elif mode in ['es_tke_', 'es__']:
                    gust_ts[hr_ind, ts_ind] = \
                                reduce_wind(kval_es[hr_ind,ts_ind], gust_es[hr_ind,ts_ind], \
                                wind_k[hr_ind,ts_ind,:], tke_k[hr_ind,ts_ind,:], \
                                model_mean[hr_ind,ts_ind], k_inds, k_dz, alpha, mode)
                elif mode in ['es_tke_rho', 'es__rho']:
                    gust_ts[hr_ind, ts_ind] = \
                                reduce_wind_rho(kval_es[hr_ind,ts_ind], gust_es[hr_ind,ts_ind], \
                                wind_k[hr_ind,ts_ind,:], tke_k[hr_ind,ts_ind,:], rho_k[hr_ind,ts_ind,:], \
                                model_mean[hr_ind,ts_ind], k_inds, k_dz, alpha, mode)
                else:
                    raise ValueError('unknown mode ' + str(mode))
        gust_hr = np.max(gust_ts, axis=1)

        print(np.mean((gust_hr-obs_gust)[0:100]))
        quit()

        deviation = gust_hr - obs_gust
        error_old = error_now
        error_now = np.sqrt(np.sum(deviation**2)/N)
        #print(error_now)
        d_error = error_now - error_old
        print('c ' + str(c) + ' alpha ' + str(np.round(alpha,5)) + ' error ' + str(error_now) + ' d_error ' + str(d_error))
        print(np.mean(gust_hr))
        print(np.mean(obs_gust))
        quit()
        #errors.append(error_now)
        #alphas.append(alpha)

        if d_error < 0:
            alpha = alpha + d_alpha


    #import matplotlib.pyplot as plt
    #plt.scatter(alphas, errors)
    #plt.xlabel('alpha')
    #plt.ylabel('error')
    #plt.show()

    print(alpha)


    gust_hr_orig = np.max(gust_lb,axis=1)
    suptitle = 'PHY PROF '
    plot_error(obs_gust, model_mean_hr, obs_mean, gust_hr, gust_hr_orig)
    plt.suptitle(suptitle + mode)
    plot_name_title = 'phys_braprof_'

    if i_plot == 1:
        plt.show()
    elif i_plot > 1:
        if i_label == '':
            plot_name = CN.plot_path + plot_name_title +str(mode)+'.png'
        else:
            plot_name = CN.plot_path + plot_name_title+str(i_label)+'_'+str(mode)+'.png'
        print(plot_name)
        plt.savefig(plot_name)
        plt.close('all')








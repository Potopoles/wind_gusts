import os
import numpy as np
import globals as G
from datetime import datetime, timedelta

def write_performance_report(CN, data_out, score_name, reverse, scores_ref):
    #print(score_name)


    scores = []
    model_keys = []
    for model_key in data_out['models'].keys():
        model_keys.append(model_key)
        scores.append(data_out['scores'][model_key][score_name])

    # add reference
    scores.append(scores_ref[score_name])
    model_keys.append('REF')

    scores, model_keys = zip(*sorted(zip(scores, model_keys), reverse=reverse))
    #print(model_keys)
    #print(scores)

    file_name = CN.output_path + score_name + '.txt'
    with open(file_name, 'w') as f:
        f.write('{}\n'.format(score_name))
        for i in range(0,len(scores)):
            f.write('{}\t {}\n'.format(model_keys[i], np.round(scores[i],2)))



#def apply_full_scaling(var):
#    mean_var = np.mean(var)
#    var = var - mean_var
#    sd_var = np.std(var)
#    var = var/sd_var
#    return(var, mean_var, sd_var)

#def box_cox(data, l1, l2):
#    if l1 != 0:
#        data = (data + l2)**l1 - 1
#    else:
#        data = np.log(data + l2)
#    return(data)


def rotate(xy, angle):
    rot_mat = np.column_stack( [[ np.cos(angle), np.sin(angle)],
                                [-np.sin(angle), np.cos(angle)]] )
    xy_rot = rot_mat @ xy
    return(xy_rot)


def fill_dict(lm, value=0):
    dict = {}
    for pred_name in lm.keys():
        dict[pred_name] = value
    return(dict)

def apply_scaling(var):
    sd_var = np.std(var)
    var = var/sd_var
    return(var, sd_var)

def find_hourly_max(gust, fields=None):
    if fields is not None: 
        # find maximum gust
        maxid = gust.argmax(axis=1)
        I = np.indices(maxid.shape)
        gust_max = gust[I,maxid].squeeze()
        max_fields = {}
        for field_name,value in fields.items():
            max_fields[field_name] = fields[field_name][I,maxid].squeeze()
        return(gust_max, max_fields)
    else:
        gust_max = np.max(gust, axis=1)
        return(gust_max)


def calc_bins(n_bins, max_val_err_space=60, weight_slope=0):
    bin_limits = np.linspace(0,max_val_err_space, n_bins+1)
    bins = {'1_1':[],'err':[]}
    weight0 = 1
    bin_weights = []
    for i in range(0,n_bins):
        bins['err'].append( (bin_limits[i],bin_limits[i+1]) )
        bins['1_1'].append( (bin_limits[i]*np.sqrt(2),
                             bin_limits[i+1]*np.sqrt(2)) )
        # weights (assuming linear trend with slope weights_slope)
        bin_weights.append( weight0 + weight_slope*i )

    #print('######################################')
    #print('bins err space: ' + str(bins['err']))
    #print('bins 1_1 space: ' + str(bins['1_1']))
    #print('bin  weights  : ' + str(bin_weights))
    #print('######################################')
    
    return(bins, bin_weights)



def prepare_model_params(use_model_fields, lm_runs, CN,
                        stat_fort_ind, stat_key, mod_stations,
                        model_params, model_dt, nhrs_forecast,
                        data_obs_stat, progress, njobs,
                        grid_point_selection):

    print('############# use ' + stat_key + '\t' + \
            str(round(progress/len(mod_stations),2)))

    # find indices for columns in model time step files
    tstep_col_ind = np.argwhere(np.asarray(model_params) == 'tstep')[0]
    zvp10_col_ind = np.argwhere(np.asarray(model_params) == 'zvp10')[0]

    i_shift_col_ind = np.argwhere(np.asarray(model_params) == 'i_shift')[0]
    j_shift_col_ind = np.argwhere(np.asarray(model_params) == 'j_shift')[0]

    # prepare station observation fields
    n_hours_all_lm = len(lm_runs)*nhrs_forecast
    obs_mean_stat  = np.full( ( n_hours_all_lm ), np.nan )
    obs_gust_stat  = np.full( ( n_hours_all_lm ), np.nan )

    ## TESTING containes mean wind values for 
    # best fitting grid point and centre grid point
    best_mean_winds = np.full(n_hours_all_lm, np.nan)
    centre_mean_winds = np.full(n_hours_all_lm, np.nan)

    # prepare station model_fields
    model_fields_stat = {}
    for field_name in use_model_fields:
        model_fields_stat[field_name] = \
                np.full( ( n_hours_all_lm, int(3600/model_dt) ), np.nan )

    # load data from all lm runs
    for lmI,lm_run in enumerate(lm_runs):

        ###################################################################
        ####### PART 1: Read and format model time step output file
        ###################################################################
        # construct file path
        mod_file_path = CN.raw_mod_path + lm_run + '/' + 'fort.' + str(stat_fort_ind)

        # time of first time step
        start_time = datetime.strptime(lm_run, '%Y%m%d%H')

        # opt 1: ignore invalid values but warn
        #values = np.genfromtxt(mod_file_path, delimiter=',',\
        #           dtype=np.float, loose=1)
        #if np.sum(np.isnan(values)) > 0:
        #    print(str(np.sum(np.isnan(values))) + ' invalid values!')
        # opt 2: raise error for invalid values
        values = np.genfromtxt(mod_file_path, delimiter=',',\
                    dtype=np.float)

        # After read in time step 0 has to be removed.
        # However, each time step has n_grid_points file rows,
        # where n_grid_points is the number
        # of output grid points around each station.
        # Therefore, first determine n_grid_points and remove the
        # n_grid_points first lines of file.
        n_grid_points = 0
        n_grid_points = np.sum(values[:,tstep_col_ind] == 0)
        if n_grid_points == 0:
            raise ValueError('n_grid_points is 0! Possibly ' + \
                                'model output is not as expected')
        values = values[n_grid_points:,:]
        time_steps = np.unique(values[:,tstep_col_ind])
        n_time_steps = len(time_steps)
        if not values.shape[0]/n_time_steps == n_grid_points:
            raise ValueError('number of entries does not ' + \
                                'divide by n_time_steps!')


###############################################################################
###############################################################################
        if grid_point_selection == 'BEST':

            ###################################################################
            ####### PART 2: For each hour find best fitting model grid point
            ###################################################################
            # 1) For each hour, find for all of the n_grid_points of model 
            #    output the corresponding 3600/dt time steps
            # 2) Calculate the model hourly mean wind. Compare to observed
            #    hourly mean wind and determine the best fitting model grid
            #    point.
            # 3) Only keep this one and store in final_values

            nts_per_hr = int(3600/model_dt)

            ### TESTING containes mean wind values for 
            ## best fitting grid point and centre grid point
            #best_mean_winds = np.full(nhrs_forecast, np.nan)
            #centre_mean_winds = np.full(nhrs_forecast, np.nan)

            # loop over hours in lm_run
            # hour label corresponds to time before label
            hrs = range(1,nhrs_forecast+1)
            for hI,hr in enumerate(hrs):
                #hr = 15

                # contains model mean wind values for each grid point and
                # time step of given hour
                model_mean_gp = np.zeros( (n_grid_points, nts_per_hr) )
                model_mean_centre = np.zeros( nts_per_hr )

                # loop over time steps of current hour
                # attention: this happens in model counter style (starting @ 1)
                #            not in python style (starting @ 0))
                ts_inds = range((hr-1)*nts_per_hr+1, hr*nts_per_hr+1)
                for tsI,ts_ind in enumerate(ts_inds):
                    row_inds = range((ts_ind-1)*n_grid_points, \
                                    ts_ind*n_grid_points)
                    model_mean_gp[:,tsI] = values[row_inds,zvp10_col_ind]
                    model_mean_centre[tsI] = values[row_inds,zvp10_col_ind]\
                          [(values[row_inds,i_shift_col_ind] == 0) & \
                           (values[row_inds,j_shift_col_ind] == 0)]

                # whenever zvp10 value is 0.000, this means that the grid point
                # was not in computation subdomain  domain and thus no value
                # available. --> set these to nan
                model_mean_gp[model_mean_gp == 0] = np.nan

                # calculate hour mean value
                model_mean_wind = np.mean(model_mean_gp, axis=1)

                # fill in observations
                cur_time = start_time + timedelta(hours=hr)
                obs_mean_hr  = data_obs_stat[G.OBS_MEAN_WIND].\
                                    loc[cur_time]
                obs_gust_hr = data_obs_stat[G.OBS_GUST_SPEED].\
                                    loc[cur_time]

                if (grid_point_selection == 'BEST') and not np.isnan(obs_mean_hr):
                    # select best fitting model grid point
                    # nans are ignored (nans are created when the grid point
                    # is not within domain decomposition subdomain)
                    grid_point_ind = np.nanargmin(np.abs(
                                        model_mean_wind - obs_mean_hr))

                # If observation is not available fall back to centre grid point
                elif (grid_point_selection == 'CENTRE') or np.isnan(obs_mean_hr):
                    # select centre model grid point
                    grid_point_ind = np.argwhere(
                                (values[row_inds,i_shift_col_ind] == 0) & \
                                   (values[row_inds,j_shift_col_ind] == 0))[0]


                # select best fitting rows with time steps of best fitting
                # grid point
                sel_inds = (np.asarray(ts_inds)-1) * n_grid_points + \
                                                    grid_point_ind

            ###################################################################
            ####### PART 3 BEST: Fill in output arrays
            ###################################################################
                hr_sel_inds = lmI*nhrs_forecast + hI
                obs_mean_stat[hr_sel_inds] = obs_mean_hr
                obs_gust_stat[hr_sel_inds] = obs_gust_hr

                # store mean wind values for separate output (testing)
                best_mean_winds[hr_sel_inds] = model_mean_wind[grid_point_ind]
                centre_mean_winds[hr_sel_inds] = np.mean(model_mean_centre)

                for field_name in use_model_fields:
                    col_ind = np.argwhere(
                                np.asarray(model_params) == field_name)[0]

                    # fill in model values
                    model_fields_stat[field_name][hr_sel_inds,:] = \
                                            values[sel_inds,col_ind]

                    # check for nan in model output.
                    # if present print warning and set observation values
                    # to nan. Like this, later scripts will not use these
                    # hour/station pairs.
                    na_sum = np.sum(np.isnan(values[sel_inds,col_ind]))
                    if na_sum > 0:
                        print('nan values in model output!!! ' + \
                                str(na_sum) + ' in ' + stat_key + \
                                ' in lmrun ' + str(lm_run) + \
                                ' in hr ' + str(hr) + \
                                ' in field name ' + str(field_name))
                        obs_mean_stat[hr_sel_inds] = np.nan
                        obs_gust_stat[hr_sel_inds] = np.nan


###############################################################################
###############################################################################
        elif grid_point_selection == 'CENTRE':

            ###################################################################
            ####### PART 3 CENTRE: Fill in output arrays
            ###################################################################
            nts_per_hr = int(3600/model_dt)

            inds = ((values[:,i_shift_col_ind] == 0) & \
                    (values[:,j_shift_col_ind] == 0)).squeeze()
            centre_values = values[inds,:]

            hrs = range(1,nhrs_forecast+1)
            for hI,hr in enumerate(hrs):
                hr_sel_inds = lmI*nhrs_forecast + hI
                ts_inds = range(hI*nts_per_hr,(hI+1)*nts_per_hr)

                # fill in observations
                cur_time = start_time + timedelta(hours=hr)
                obs_mean_hr  = data_obs_stat[G.OBS_MEAN_WIND].\
                                    loc[cur_time]
                obs_gust_hr = data_obs_stat[G.OBS_GUST_SPEED].\
                                    loc[cur_time]
                obs_mean_stat[hr_sel_inds] = obs_mean_hr
                obs_gust_stat[hr_sel_inds] = obs_gust_hr

                # fill in model values
                for field_name in use_model_fields:
                    col_ind = np.argwhere(
                                np.asarray(model_params) == field_name)[0]

                    # fill in model values
                    model_fields_stat[field_name][hr_sel_inds,:] = \
                                            centre_values[ts_inds,col_ind]


    result = (  obs_mean_stat,
                obs_gust_stat,
                model_fields_stat,
                best_mean_winds,
                centre_mean_winds
             )
    return(result)







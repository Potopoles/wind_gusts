import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import globals as G
from namelist_cases import Case_Namelist
import namelist_cases as nl
import os
from netCDF4 import Dataset

############ USER INPUT #############
case_index = nl.case_index
CN = Case_Namelist(case_index)
# time step [s] of model
model_dt = 10
model_integ_n_hrs = 24
# starting index of fortran files
ind0 = 701

# header of fortran output files
if CN.exp_id == 101:
    model_params = \
        ['tstep','i_shift','j_shift',
         'k_bra_es','k_bra_lb','k_bra_ub', # time step and model levels of brassuer         # 3
         'tcm','zvp10', # turbulent coefficient of momentum and abs wind at 10 m            # 5 
         'zv_bra_es','zv_bra_lb','zv_bra_ub', # brasseur gust velocities                    # 8
         'ul1', 'vl1', # u and v at lowest model level                                      # 13
         'tkel1',# 'tke_bra_es', # tke at lowest level and mean tke between sfc and bra estimate # 14
         'z0', 'Tl1', # surface roughness and temperature at lowest model level             # 16
         'shflx', 'qvflx', # surface sensible heat and water vapor flux 
         'Tskin', 'qvl1', # skin temperature and water vapor at lowest model level
         'phil1', # geopotential at lowest model level 
         'ps'] # surface pressure
else:
    raise NotImplementedError()


hist_tag = '02_prep_model'
sso_stdh_file = '../extern_par/SSO_STDH.nc'
slo_ang_file = '../extern_par/SLO_ANG.nc'
skyview_file = '../extern_par/SKYVIEW.nc'
slo_asp_file = '../extern_par/SLO_ASP.nc'
z0_file = '../extern_par/Z0.nc'
hsurf_file = '../extern_par/HSURF.nc'
#####################################

lm_runs = os.listdir(CN.raw_mod_path)
mod_stations_file = CN.raw_mod_path + lm_runs[0] + '/fort.700'
print(mod_stations_file)

# read model station names
mod_stations = np.genfromtxt(mod_stations_file, skip_header=2, dtype=np.str)[:,0]
file_inds = ind0 + np.arange(0,len(mod_stations))
#file_inds_2 = ind0_2 + np.arange(0,len(mod_stations))
stat_i_inds = np.genfromtxt(mod_stations_file, skip_header=2, dtype=np.str)[:,8].astype(np.int)

if case_index == 0:
    use_stat = file_inds <= 731
    # Filter out stations with i_ind = 0 (those with height = -99999999)
    use_stat[stat_i_inds == 0] = False
elif case_index == 9:
    use_stat = file_inds <= 911
    # Filter out stations with i_ind = 0 (those with height = -99999999)
    use_stat[stat_i_inds == 0] = False
    #use_stat = stat_i_inds != 0
else:
    # Filter out stations with i_ind = 0 (those with height = -99999999)
    use_stat = stat_i_inds != 0


# filter out missing files
exist_file_inds = [int(file[5:])-ind0 for file in os.listdir(CN.raw_mod_path + lm_runs[0])]
exist_file_inds = [ind for ind in exist_file_inds if ind >= 0]
use_stat = [use_stat[i] if i in exist_file_inds else False for i in range(0,len(use_stat))]

mod_stations = mod_stations[use_stat]
file_inds = file_inds[use_stat]
stat_i_inds = stat_i_inds[use_stat]
stat_j_inds = np.genfromtxt(mod_stations_file, skip_header=2, dtype=np.str)[use_stat,9].astype(np.int)
stat_dz = np.genfromtxt(mod_stations_file, skip_header=2, dtype=np.str)[use_stat,11].astype(np.float)

# nc file for sso_stdh
sso_stdh = Dataset(sso_stdh_file, 'r')['SSO_STDH'][:]
#slo_ang = Dataset(slo_ang_file, 'r')['SLO_ANG'][:]
#skyview = Dataset(skyview_file, 'r')['SKYVIEW'][:]
#slo_asp = Dataset(slo_asp_file, 'r')['SLO_ASP'][:]
z0 = Dataset(z0_file, 'r')['Z0'][:]
hsurf = Dataset(hsurf_file, 'r')['HSURF'][:]

# load main data file
data = pd.read_pickle(CN.obs_path)
# stations to use as given by observation data set
obs_stations = list(data[G.OBS][G.STAT].keys())

# add entry for model data
data[G.MODEL] = {G.STAT:{}}

# find indices for columns in model time step files
tstep_col_ind = np.argwhere(np.asarray(model_params) == 'tstep')[0]
zvp10_col_ind = np.argwhere(np.asarray(model_params) == 'zvp10')[0]
#i_shift_col_ind = np.argwhere(np.asarray(model_params) == 'i_shift')[0]
#j_shift_col_ind = np.argwhere(np.asarray(model_params) == 'j_shift')[0]

## TODO
n_hours_all_lm = len(lm_runs)*model_integ_n_hrs
#zvp10 = np.zeros( ( n_hours_all_lm, 1, int(3600/model_dt) ) )

# read model data
for i,stat_key in enumerate(mod_stations):
    ind = file_inds[i]
    print(ind)
    #ind_2 = file_inds_2[i]

    if stat_key in obs_stations:
        print('use ' + stat_key)

        ## Add model related information to station metadata
        #series = pd.Series([stat_dz[i]], index=data[G.STAT_META][stat_key].index)
        #data[G.STAT_META][stat_key]['dz'] = series

        series = pd.Series(sso_stdh[stat_j_inds[i],stat_i_inds[i]],
                        index=data[G.STAT_META][stat_key].index)
        data[G.STAT_META][stat_key]['sso_stdh'] = series

        #series = pd.Series(slo_ang[stat_j_inds[i],stat_i_inds[i]],
        #                index=data[G.STAT_META][stat_key].index)
        #data[G.STAT_META][stat_key]['slo_ang'] = series

        #series = pd.Series(skyview[stat_j_inds[i],stat_i_inds[i]],
        #                index=data[G.STAT_META][stat_key].index)
        #data[G.STAT_META][stat_key]['skyview'] = series

        #series = pd.Series(slo_asp[stat_j_inds[i],stat_i_inds[i]],
        #                index=data[G.STAT_META][stat_key].index)
        #data[G.STAT_META][stat_key]['slo_asp'] = series

        series = pd.Series(z0[stat_j_inds[i],stat_i_inds[i]],
                        index=data[G.STAT_META][stat_key].index)
        data[G.STAT_META][stat_key]['z0'] = series

        series = pd.Series(hsurf[stat_j_inds[i],stat_i_inds[i]],
                        index=data[G.STAT_META][stat_key].index)
        data[G.STAT_META][stat_key]['hsurf'] = series


        # add model data
        raw_data = {}
        #raw_data2 = {}
        stat_data = {G.RAW:raw_data}
        #stat_data = {G.RAW:raw_data, G.RAW2:raw_data2}
        data[G.MODEL][G.STAT][stat_key] = stat_data

        # load data from all lm runs
        for lm_run in lm_runs:

            ###################################################################
            ########### PART 1: Read and format model time step output file
            ###################################################################
            # construct file path
            mod_file_path = CN.raw_mod_path + lm_run + '/' + 'fort.' + str(ind)

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


            ###################################################################
            ########### PART 2: TODO
            ###################################################################
            # 1) For each hour, find for all of the n_grid_points of model 
            #    output the corresponding 3600/dt time steps
            # 2) Calculate the model hourly mean wind. Compare to observed
            #    hourly mean wind and determine the best fitting model grid
            #    point.
            # 3) Only keep this one and store in final_values

            nts_per_hr = int(3600/model_dt)
            final_values = np.full( (n_hours_all_lm*nts_per_hr, values.shape[1]), np.nan )

            # TODO remove this after testing
            # contains for each hour the index of the best fitting model
            # grid point
            best_fit_gp_inds = np.full(model_integ_n_hrs, np.nan)
            best_mean_winds = np.full(model_integ_n_hrs, np.nan)

            # loop over hours in lm_run
            # hour label corresponds to time before label
            hrs = range(1,model_integ_n_hrs+1)
            for hI,hr in enumerate(hrs):
                #hr = 15

                # contains model mean wind values for each grid point and
                # time step of given hour
                model_mean_gp = np.zeros( (n_grid_points, nts_per_hr) )

                # loop over time steps of current hour
                # attention: this happens in model counter style starting @ 1)
                #            not in python style (starting @ 0))
                ts_inds = range((hr-1)*nts_per_hr+1, hr*nts_per_hr+1)
                for i,ts_ind in enumerate(ts_inds):
                    row_inds = range((ts_ind-1)*n_grid_points, \
                                    ts_ind*n_grid_points)
                    model_mean_gp[:,i] = values[row_inds,zvp10_col_ind]
                model_mean_wind = np.mean(model_mean_gp, axis=1)

                #plt.plot(model_mean_gp.T)
                #plt.axhline(y=obs_mean_wind, color='k', linewidth=3)
                #plt.show()

                # get observation date
                cur_time = start_time + timedelta(hours=hr)
                obs_mean_wind = data[G.OBS][G.STAT]['ABO'][G.OBS_MEAN_WIND].\
                                    loc[cur_time]

                # select best fitting model grid point
                grid_point_ind = np.argmin(np.abs(
                                    model_mean_wind - obs_mean_wind))
                # TODO testing
                best_fit_gp_inds[hI] = grid_point_ind
                best_mean_winds[hI] = model_mean_wind[grid_point_ind]

                # select best fitting rows with time steps of best fitting
                # grid point
                sel_inds = (np.asarray(ts_inds)-1) * n_grid_points + \
                                                    grid_point_ind
                sel_values = values[sel_inds,:]

                ## TODO testing
                #print(np.mean(sel_values[:,zvp10_col_ind]))

                # store selection of current hour in final value array for 
                # this lm_run
                final_values[hI*nts_per_hr:(hI+1)*nts_per_hr, :] = sel_values

            # Save in raw_data dictionary
            n_entries = values.shape[0]
            ts_secs = (np.arange(1,model_integ_n_hrs*nts_per_hr+1)*model_dt).\
                        astype(np.float)
            dts = [start_time + timedelta(seconds=ts_sec) for ts_sec in ts_secs]
            df = pd.DataFrame(final_values, index=dts, columns=model_params)
            #df = df.resample('D').max()
            raw_data[lm_run] = df

    else:
        print('do not use ' + stat_key)

print('##################')

# save names of used stations
data[G.STAT_NAMES] = list(data[G.MODEL][G.STAT].keys())

data[G.HIST].append(hist_tag)

# save output file
pickle.dump(data, open(CN.new_mod_path, 'wb'))

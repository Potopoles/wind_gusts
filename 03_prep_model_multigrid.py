import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import globals as G
from namelist_cases import Case_Namelist
import namelist_cases as nl
import pickle, os, sys
from netCDF4 import Dataset

from tuning_functions import (prepare_model_params)
from plot_functions import (draw_error_percentile_lines,
                            draw_error_grid,
                            get_point_col)
import multiprocessing as mp

############ USER INPUT #############
case_index = nl.case_index
CN = Case_Namelist(case_index)
# time step [s] of model
model_dt = nl.model_dt
nhrs_forecast = nl.nhrs_forecast
# starting index of fortran files
ind0 = 701
debug_min_stat_ind = 701
debug_max_stat_ind = 1001
i_draw_grid_wind_plot = 1
grid_point_selection = nl.grid_point_selection

njobs = 1
if len(sys.argv) > 1:
    njobs = int(sys.argv[1])
    print('Number of jobs is set to ' + str(njobs) + '.')
else:
    print('Number of jobs not given. Default value is ' + str(njobs) + '.')

use_model_fields = ['k_bra_es','k_bra_lb','k_bra_ub',
                    'tcm', 'zvp10',
                    'zv_bra_es','zv_bra_lb','zv_bra_ub',
                    'ul1', 'vl1', 'tkel1', 'z0', 'Tl1',
                    'shflx', 'qvflx', 'Tskin', 'qvl1','phil1', 'ps']

# header of fortran output files
if CN.exp_id == 101:
    model_params = \
    ['tstep','i_shift','j_shift',
     'k_bra_es','k_bra_lb','k_bra_ub', # time step and model levels of brassuer
     'tcm','zvp10', # turbulent coefficient of momentum and abs wind at 10 m
     'zv_bra_es','zv_bra_lb','zv_bra_ub', # brasseur gust velocities
     'ul1', 'vl1', # u and v at lowest model level          
     'tkel1',# tke at lowest level
     'z0', 'Tl1', # surface roughness and temperature at lowest model level
     'shflx', 'qvflx', # surface sensible heat and water vapor flux 
     'Tskin', 'qvl1', # skin temperature and water vapor at lowest model level
     'phil1', # geopotential at lowest model level 
     'ps'] # surface pressure
else:
    raise NotImplementedError()

hist_tag = '02_prep_model'

ext_files = {}
ext_files['sso_stdh']   = '../extern_par/SSO_STDH.nc'
#ext_files['slo_ang']    = '../extern_par/SLO_ANG.nc'
#ext_files['skyview']    = '../extern_par/SKYVIEW.nc'
#ext_files['slo_asp']    = '../extern_par/SLO_ASP.nc'
ext_files['z0']         = '../extern_par/Z0.nc'
ext_files['hsurf']      = '../extern_par/HSURF.nc'
#####################################


###########################################################################
############### PART 0: PREPARATIONS
###########################################################################

lm_runs = os.listdir(CN.raw_mod_path)
mod_stations_file = CN.raw_mod_path + lm_runs[0] + '/fort.700'
print(mod_stations_file)

# read model station names
mod_stations = np.genfromtxt(mod_stations_file,
                skip_header=2, dtype=np.str)[:,0]
file_inds = ind0 + np.arange(0,len(mod_stations))
stat_i_inds = np.genfromtxt(mod_stations_file, 
                skip_header=2, dtype=np.str)[:,8].astype(np.int)

if case_index == 0:
    use_stat = (file_inds <= debug_max_stat_ind) & (file_inds >= debug_min_stat_ind)
    # Filter out stations with i_ind = 0
    # (those with height = -99999999)
    use_stat[stat_i_inds == 0] = False
else:
    # Filter out stations with i_ind = 0
    # (those with height = -99999999)
    use_stat = stat_i_inds != 0

# TODO: DEBUG
#use_stat = file_inds <= debug_max_stat_ind


# filter out missing files
exist_file_inds = [int(file[5:])-ind0 for file in \
                    os.listdir(CN.raw_mod_path + lm_runs[0])]
exist_file_inds = [ind for ind in exist_file_inds if ind >= 0]
use_stat = [use_stat[i] if i in exist_file_inds else False \
                        for i in range(0,len(use_stat))]

mod_stations = mod_stations[use_stat]
file_inds = file_inds[use_stat]
stat_i_inds = stat_i_inds[use_stat]
stat_j_inds = np.genfromtxt(mod_stations_file, skip_header=2,
                    dtype=np.str)[use_stat,9].astype(np.int)
stat_dz = np.genfromtxt(mod_stations_file, skip_header=2,
                    dtype=np.str)[use_stat,11].astype(np.float)

# nc file for sso_stdh
sso_stdh = Dataset(ext_files['sso_stdh'], 'r')['SSO_STDH'][:]
#slo_ang = Dataset(ext_files['slo_ang'],  'r')['SLO_ANG'][:]
#skyview = Dataset(ext_files['skyview'],  'r')['SKYVIEW'][:]
#slo_asp = Dataset(ext_files['slo_asp'],  'r')['SLO_ASP'][:]
z0       = Dataset(ext_files['z0'],       'r')['Z0'][:]
hsurf    = Dataset(ext_files['hsurf'],    'r')['HSURF'][:]

# load main data file
data = pd.read_pickle(CN.obs_path)
# stations to use as given by observation data set
obs_stations = list(data[G.OBS][G.STAT].keys())

## add entry for model data
#data[G.MODEL] = {G.STAT:{}}

# set up containers for model_fields
model_fields = {}
n_hours_all_lm = len(lm_runs)*nhrs_forecast
obs_mean   = np.full( ( n_hours_all_lm, len(mod_stations) ), np.nan )
obs_gust   = np.full( ( n_hours_all_lm, len(mod_stations) ), np.nan )
for field_name in use_model_fields:
    model_fields[field_name]  = np.full( ( n_hours_all_lm,
                                           len(mod_stations), 
                                           int(3600/model_dt) ), np.nan )
best_model_mean   = np.full( ( n_hours_all_lm, len(mod_stations) ), np.nan )
centre_model_mean = np.full( ( n_hours_all_lm, len(mod_stations) ), np.nan )


###########################################################################
############### PART 1: STATION META DATA
###########################################################################
data[G.STAT_NAMES] = []
for sI,stat_key in enumerate(mod_stations):
    stat_fort_ind = file_inds[sI]

    if stat_key in obs_stations:

        data[G.STAT_NAMES].append(stat_key)

        ## Add model related information to station metadata
        #series = pd.Series([stat_dz[sI]], index=data[G.STAT_META][stat_key].index)
        #data[G.STAT_META][stat_key]['dz'] = series

        series = pd.Series(sso_stdh[stat_j_inds[sI],stat_i_inds[sI]],
                        index=data[G.STAT_META][stat_key].index)
        data[G.STAT_META][stat_key]['sso_stdh'] = series

        #series = pd.Series(slo_ang[stat_j_inds[[sI]],stat_i_inds[sI]],
        #                index=data[G.STAT_META][stat_key].index)
        #data[G.STAT_META][stat_key]['slo_ang'] = series

        #series = pd.Series(skyview[stat_j_inds[[sI]],stat_i_inds[[sI]]],
        #                index=data[G.STAT_META][stat_key].index)
        #data[G.STAT_META][stat_key]['skyview'] = series

        #series = pd.Series(slo_asp[stat_j_inds[[sI]],stat_i_inds[[sI]]],
        #                index=data[G.STAT_META][stat_key].index)
        #data[G.STAT_META][stat_key]['slo_asp'] = series

        series = pd.Series(z0[stat_j_inds[[sI]],stat_i_inds[[sI]]],
                        index=data[G.STAT_META][stat_key].index)
        data[G.STAT_META][stat_key]['z0'] = series

        series = pd.Series(hsurf[stat_j_inds[[sI]],stat_i_inds[[sI]]],
                        index=data[G.STAT_META][stat_key].index)
        data[G.STAT_META][stat_key]['hsurf'] = series

    else:
        data[G.STAT_NAMES].append(np.nan)

data[G.STAT_NAMES] = np.asarray(data[G.STAT_NAMES])


#obs_stations = ['NABDUE']

###########################################################################
############### PART 2: PREPARE MODEL DATA
###########################################################################
if njobs == 1:
    for sI,stat_key in enumerate(mod_stations):
        stat_fort_ind = file_inds[sI]
        #print(stat_fort_ind)

        if stat_key in obs_stations:

            result = prepare_model_params(use_model_fields, lm_runs, CN,
                                stat_fort_ind, stat_key, mod_stations,
                                model_params, model_dt, nhrs_forecast,
                                data[G.OBS][G.STAT][stat_key], sI, njobs,
                                grid_point_selection)
            obs_mean[:,sI] = np.nan
            obs_gust[:,sI]  = np.nan
            model_fields_stat = result[2]
            for field_name in use_model_fields:
                model_fields[field_name][:,sI,:] = \
                                model_fields_stat[field_name]
            best_model_mean[:,sI]   = result[3]
            centre_model_mean[:,sI] = result[4]
        else:
            print('############# do not use ' + stat_key)

# if njobs > 1 will run in parallel over stations
elif njobs > 1:

    p = mp.Pool(processes=njobs)
    input = []

    for sI,stat_key in enumerate(mod_stations):
        stat_fort_ind = file_inds[sI]

        if stat_key in obs_stations:
            input.append((use_model_fields, lm_runs, CN,
                        stat_fort_ind, stat_key, mod_stations,
                        model_params, model_dt, nhrs_forecast,
                        data[G.OBS][G.STAT][stat_key], sI, njobs,
                        grid_point_selection))
        else:
            print('############# do not use ' + stat_key)


    result = p.starmap(prepare_model_params, input)
    p.close()
    p.join()

    c = 0
    for sI,stat_key in enumerate(mod_stations):
        stat_fort_ind = file_inds[sI]

        if stat_key in obs_stations:
            obs_mean[:,sI]  = result[c][0]
            obs_gust[:,sI]  = result[c][1]
            model_fields_stat = result[c][2]
            for field_name in use_model_fields:
                model_fields[field_name][:,sI,:] = \
                                model_fields_stat[field_name]
            best_model_mean[:,sI]   = result[c][3]
            centre_model_mean[:,sI] = result[c][4]

            c += 1


###########################################################################
############### PART 3: SAVE DATA
###########################################################################
print('##################')

## save names of used stations
#data[G.STAT_NAMES] = list(data[G.MODEL][G.STAT].keys())

data[G.HIST].append(hist_tag)

#print(sys.getsizeof(obs_mean)/1000000)
#print(sys.getsizeof(obs_gust)/1000000)
#size = 0
#for field in model_fields.keys():
#    size += sys.getsizeof(model_fields[field])
#print(size/1000000)
#print(sys.getsizeof(best_model_mean)/1000000)
#print(sys.getsizeof(centre_model_mean)/1000000)


data['obs_mean'] = obs_mean
data['obs_gust'] = obs_gust
data['best_model_mean'] = best_model_mean
data['centre_model_mean'] = centre_model_mean

# Store model output fields in nc file.
ncf = Dataset(CN.mod_nc_path, 'w')
shape = model_fields['zvp10'].shape
ncf.createDimension('hour', size=shape[0])
ncf.createDimension('station', size=shape[1])
ncf.createDimension('ts', size=shape[2])
for field_name in model_fields.keys():
    var = ncf.createVariable(field_name, 'f', ('hour','station','ts'))
    var[:] = model_fields[field_name]
ncf.close()

# save output file
pickle.dump(data, open(CN.mod_path, 'wb'))


###########################################################################
############### PART 4: DRAW PLOT
###########################################################################
if (i_draw_grid_wind_plot and (grid_point_selection == 'BEST')):
    # create directories
    if not os.path.exists(CN.plot_path):
        os.mkdir(CN.plot_path)

    limit = 35

    # process data
    best_model_mean = best_model_mean.flatten()
    best_model_mean = best_model_mean[np.isnan(best_model_mean) == False]
    centre_model_mean = centre_model_mean.flatten()
    centre_model_mean = centre_model_mean[np.isnan(centre_model_mean) == False]
    print(best_model_mean.shape)
    print(centre_model_mean.shape)
    # draw plot
    plt.scatter(centre_model_mean, best_model_mean,
                marker=".",
                color=get_point_col(centre_model_mean, best_model_mean))
    ax = plt.gca()
    ax.set_xlim((0,limit))
    ax.set_ylim((0,limit))
    ax.set_xlabel('centre grid point')
    ax.set_ylabel('best fitting grid point')
    #ax.grid()
    draw_error_grid(limit, limit, ax)
    ax.set_aspect('equal')
    draw_error_percentile_lines(centre_model_mean, best_model_mean, ax, rot_angle=-np.pi/4)
    #plt.show()
    plot_name = CN.plot_path + 'grid_point_wind.png'
    plt.savefig(plot_name)

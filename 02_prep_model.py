import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import globals as G
from namelist_cases import Case_Namelist
import os
from netCDF4 import Dataset

############ USER INPUT #############
case_index = 15
CN = Case_Namelist(case_index)
# time step [s] of model
model_dt = 10
# starting index of fortran files
ind0 = 701
#ind0_2 = 5701
# header of fortran output files
model_params = ['ntstep','k_bra_es','k_bra_lb','k_bra_ub', # time step and model levels of brassuer
                'tcm','zvp10', # turbulent coefficient of momentum and abs wind at 10 m
                'zv_bra_es','zv_bra_lb','zv_bra_ub', # brasseur gust velocities
                'uvl1','uvl2','uvl3', # abs wind at lowest 3 model levels
                'ul1', 'vl1', # u and v at lowest model level
                'tkel1',# 'tke_bra_es', # tke at lowest level and mean tke between sfc and bra estimate
                'z0', 'Tl1', # surface roughness and temperature at lowest model level
                'shflx', 'qvflx', # surface sensible heat and water vapor flux 
                'Tskin', 'qvl1', # skin temperature and water vapor at lowest model level
                'rho_bra_es','rho_bra_lb','rho_bra_ub', # brasseur density
                'rhol1', # rho at lowest model level
                'phil1', # geopotential at lowest model level 
                'ps']
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
stat_i_inds = np.genfromtxt(mod_stations_file, skip_header=2, dtype=np.str)[:,9].astype(np.int)

if case_index == 0:
    use_stat = file_inds <= 731
    # Filter out stations with i_ind = 0 (thos with height = -99999999)
    use_stat[stat_i_inds == 0] = False
elif case_index == 9:
    use_stat = file_inds <= 911
    # Filter out stations with i_ind = 0 (thos with height = -99999999)
    use_stat[stat_i_inds == 0] = False
    #use_stat = stat_i_inds != 0
else:
    # Filter out stations with i_ind = 0 (thos with height = -99999999)
    use_stat = stat_i_inds != 0


mod_stations = mod_stations[use_stat]
file_inds = file_inds[use_stat]
#file_inds_2 = file_inds_2[use_stat]
stat_i_inds = stat_i_inds[use_stat]
stat_j_inds = np.genfromtxt(mod_stations_file, skip_header=2, dtype=np.str)[use_stat,8].astype(np.int)
stat_dz = np.genfromtxt(mod_stations_file, skip_header=2, dtype=np.str)[use_stat,11].astype(np.float)

# nc file for sso_stdh
#sso_stdh = Dataset(sso_stdh_file, 'r')['SSO_STDH'][:]
#slo_ang = Dataset(slo_ang_file, 'r')['SLO_ANG'][:]
#skyview = Dataset(skyview_file, 'r')['SKYVIEW'][:]
#slo_asp = Dataset(slo_asp_file, 'r')['SLO_ASP'][:]
#z0 = Dataset(z0_file, 'r')['Z0'][:]
hsurf = Dataset(hsurf_file, 'r')['HSURF'][:]

# load main data file
data = pd.read_pickle(CN.obs_path)
# stations to use as given by observation data set
obs_stations = list(data[G.OBS][G.STAT].keys())

# add entry for model data
data[G.MODEL] = {G.STAT:{}}


# read model data
for i,stat_key in enumerate(mod_stations):
    ind = file_inds[i]
    #ind_2 = file_inds_2[i]

    if stat_key in obs_stations:
        print('use ' + stat_key)

        ## Add model related information to station metadata
        #series = pd.Series([stat_dz[i]], index=data[G.STAT_META][stat_key].index)
        #data[G.STAT_META][stat_key]['dz'] = series

        #series = pd.Series(sso_stdh[stat_i_inds[i],stat_j_inds[i]],
        #                index=data[G.STAT_META][stat_key].index)
        #data[G.STAT_META][stat_key]['sso_stdh'] = series

        #series = pd.Series(slo_ang[stat_i_inds[i],stat_j_inds[i]],
        #                index=data[G.STAT_META][stat_key].index)
        #data[G.STAT_META][stat_key]['slo_ang'] = series

        #series = pd.Series(skyview[stat_i_inds[i],stat_j_inds[i]],
        #                index=data[G.STAT_META][stat_key].index)
        #data[G.STAT_META][stat_key]['skyview'] = series

        #series = pd.Series(slo_asp[stat_i_inds[i],stat_j_inds[i]],
        #                index=data[G.STAT_META][stat_key].index)
        #data[G.STAT_META][stat_key]['slo_asp'] = series

        #series = pd.Series(z0[stat_i_inds[i],stat_j_inds[i]],
        #                index=data[G.STAT_META][stat_key].index)
        #data[G.STAT_META][stat_key]['z0'] = series

        series = pd.Series(hsurf[stat_i_inds[i],stat_j_inds[i]],
                        index=data[G.STAT_META][stat_key].index)
        data[G.STAT_META][stat_key]['hsurf'] = series


        # add model data
        raw_data = {}
        raw_data2 = {}
        stat_data = {G.RAW:raw_data, G.RAW2:raw_data2}
        data[G.MODEL][G.STAT][stat_key] = stat_data

        # load data from all lm runs
        for lm_run in lm_runs:
            #print(lm_run)

            # construct file path
            mod_file_path = CN.raw_mod_path + lm_run + '/' + 'fort.' + str(ind)
            #mod_file_path_2 = CN.raw_mod_path + lm_run + '/' + 'fort.' + str(ind_2)

            # tiem of first time step
            start_time = datetime.strptime(lm_run, '%Y%m%d%H')

            values = np.genfromtxt(mod_file_path, delimiter=',', dtype=np.float, loose=1)[:-1]
            #values_2 = np.genfromtxt(mod_file_path_2, delimiter=',', dtype=np.float, loose=1)[:-1]
            #print(np.sum(values_2 == -1))
            #print(np.sum(np.isnan(values_2)))
            #values_2[values_2 == -1] = np.nan
            if np.sum(np.isnan(values)) > 0:
                print(str(np.sum(np.isnan(values))) + ' invalid values!')
            n_entries = values.shape[0]
            ts_secs = (np.arange(0,n_entries)*10).astype(np.float)
            dts = [start_time + timedelta(seconds=ts_sec) for ts_sec in ts_secs]
            df = pd.DataFrame(values, index=dts, columns=model_params)
            #df = df.resample('D').max()
            raw_data[lm_run] = df

            #profile = values_2[:,1:] 
            #raw_data2[lm_run] = profile

    else:
        print('do not use ' + stat_key)

print('##################')

# save names of used stations
data[G.STAT_NAMES] = list(data[G.MODEL][G.STAT].keys())

data[G.HIST].append(hist_tag)

# save output file
pickle.dump(data, open(CN.mod_path, 'wb'))

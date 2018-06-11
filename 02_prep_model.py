import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import globals as G
from namelist_cases import Case_Namelist
import os

############ USER INPUT #############
case_index = 0
CN = Case_Namelist(case_index)
# time step [s] of model
model_dt = 10
# starting index of fortran files
ind0 = 701
# header of fortran output files
model_params = ['ntstep','k_bra_es','k_bra_lb','k_bra_ub','tcm','zvp10',
                'zv_bra_es','zv_bra_lb','zv_bra_ub',
                'vl1','vl2','vl3','vl4','vl5',
                'tkel1','tkel2','tkel3','tkel4','tkel5',
                'dpsdt','tke_bra_es','ps']
#####################################

lm_runs = os.listdir(CN.raw_mod_path)

mod_stations_file = CN.raw_mod_path + lm_runs[0] + '/fort.700'

# read model station names
mod_stations = np.genfromtxt(mod_stations_file, skip_header=2, dtype=np.str)[:,0]
#mod_stations = ['ABO','AEG'] # debug
mod_stations = mod_stations[:10] # debug

# load main data file
data = pd.read_pickle(CN.obs_path)
# stations to use as given by observation data set
obs_stations = list(data[G.OBS][G.STAT].keys())

# add entry for model data
data[G.MODEL] = {G.STAT:{}}

# read model data
for i,_ in enumerate(mod_stations):
    ind = i+ind0 
    #print(ind)

    stat_key = mod_stations[i]

    if stat_key in obs_stations:
        print('use ' + stat_key)
        raw_data = {}
        stat_data = {G.RAW:raw_data}
        data[G.MODEL][G.STAT][stat_key] = stat_data

        # load data from all lm runs
        for lm_run in lm_runs:

            # construct file path
            mod_file_path = CN.raw_mod_path + lm_run + '/' + 'fort.' + str(ind)

            # tiem of first time step
            start_time = datetime.strptime(lm_run, '%Y%m%d%H')

            values = np.genfromtxt(mod_file_path, delimiter=',', dtype=np.float, loose=1)[:-1]
            if np.sum(np.isnan(values)) > 0:
                print(str(np.sum(np.isnan(values))) + ' invalid values!')
            n_entries = values.shape[0]
            ts_secs = (np.arange(0,n_entries)*10).astype(np.float)
            dts = [start_time + timedelta(seconds=ts_sec) for ts_sec in ts_secs]
            df = pd.DataFrame(values, index=dts, columns=model_params)
            #df = df.resample('D').max()

            raw_data[lm_run] = df

    else:
        print('do not use ' + stat_key)
print('##################')

# save names of used stations
data[G.STAT_NAMES] = list(data[G.MODEL][G.STAT].keys())

# save output file
pickle.dump(data, open(CN.mod_path, 'wb'))

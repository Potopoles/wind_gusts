import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

############ USER INPUT #############
obs_path = '../obs_out/'
obs_inp_file = '20180103sfc_2262.'
# obs case name (name of obs pkl file in data folder)
obs_case_name = 'burglind'
out_pickle_file_path = '../data/OBS_'+obs_case_name+'.pkl'
MISSING_VALUE = -9999
sample_rate = '1H'
OBS = 'obs'
STAT = 'stations'
PAR = 'params'
# path to file containing meta information about stations (mainly column 'Use' is interesting)
stations_meta_file_path = '../obs_out/ps_fkl010b1_2262.csv'
### use_params
# VMAX_10M1: hourly max gust
# FF_10M   : 10min mean wind speed @10m
# DD_10M   : 10min mean wind direction @10m
obs_params = ['VMAX_10M1', 'FF_10M', 'DD_10M']
###
#####################################


# load data
obs_inp_file_path = obs_path + obs_inp_file
inp_all_contents = np.genfromtxt(obs_inp_file_path, skip_header=17, dtype=np.str)
inp_params = inp_all_contents[1:,0]
inp_all_contents = inp_all_contents[:,1:]
inp_station_names = inp_all_contents[0,5:]
inp_values = inp_all_contents[1:,5:].astype(np.float)
inp_values[inp_values == MISSING_VALUE] = np.nan
inp_time_ints = inp_all_contents[1:,:5].astype(np.int)
inp_dts = [datetime(dt[0],dt[1],dt[2],dt[3],dt[4]) for dt in inp_time_ints]

station_meta = pd.read_csv(stations_meta_file_path, encoding='ISO-8859-1',
                            error_bad_lines=False, sep=';')
stations_meta_use = station_meta[station_meta['Use'] == 'y']

# temporary dictionary with parameters separated
tmp = {}
for param in obs_params:
    mask_inds = np.argwhere(inp_params == param).squeeze()
    values = inp_values[mask_inds]
    dts = [inp_dts[i] for i in mask_inds]
    # create data frame
    data = pd.DataFrame(values, index=dts, columns=inp_station_names)
    # resample to corret sample_rate
    data = data.resample(sample_rate).mean()
    tmp[param] = data


# Create dictionary with each parameter as a key and pandas df as value
obs = {}
obs['dts'] = tmp[obs_params[0]].index.values.astype('M8[s]').astype('O')
obs['param_names'] = obs_params
stations = {}
for stat_key in inp_station_names: 
    if np.any(stations_meta_use['ABBR'] == stat_key):
        stations[stat_key] = {PAR:{}}
        stations[stat_key]['meta'] = stations_meta_use[stations_meta_use['ABBR'] == stat_key]
        for param in obs_params:
            stations[stat_key][PAR][param] = tmp[param][stat_key]

obs[STAT] = stations
data = {}
data[OBS] = obs

# Save
pickle.dump(data, open(out_pickle_file_path, 'wb'))

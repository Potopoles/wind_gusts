import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import globals as G
from namelist_cases import Case_Namelist

############ USER INPUT #############
case_index = 13
CN = Case_Namelist(case_index)
MISSING_VALUE = -9999
sample_rate = '1H'
### use_params
# VMAX_10M1: hourly max gust
# FF_10M   : 10min mean wind speed @10m
# DD_10M   : 10min mean wind direction @10m
#obs_params = {'VMAX_10M1': G.OBS_GUST_SPEED,
#                'FF_10M' : G.OBS_MEAN_WIND,
#                'DD_10M' : G.OBS_WIND_DIR}
obs_params = {'VMAX_10M1': G.OBS_GUST_SPEED,
                'FF_10M' : G.OBS_MEAN_WIND}
hist_tag = '01_prep_obs'
#####################################

# meta data about stations (contains information about which should be used)
station_meta = pd.read_csv(CN.stations_meta_path, encoding='ISO-8859-1',
                            error_bad_lines=False, sep=';')
stations_meta_use = station_meta[station_meta['Use'] == 'y']

# load obs data day by day
day_data = []
for obs_path in CN.raw_obs_path:
    print(obs_path)

    inp_all_contents = np.genfromtxt(obs_path, skip_header=17, dtype=np.str)
    inp_params = inp_all_contents[1:,0]
    inp_all_contents = inp_all_contents[:,1:]
    inp_station_names = inp_all_contents[0,5:]
    inp_values = inp_all_contents[1:,5:].astype(np.float)
    inp_values[inp_values == MISSING_VALUE] = np.nan
    inp_time_ints = inp_all_contents[1:,:5].astype(np.int)
    inp_dts = [datetime(dt[0],dt[1],dt[2],dt[3],dt[4]) for dt in inp_time_ints]

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
        
        # remove weird extremly large values
        data[(data.values > 150) | (data.values < 0)] = np.nan
        #badsum =np.sum(data.values > 1000) 
        #if badsum:
        #    print(badsum)
        #    print(data.values[data.values > 1000])
        #quit()
        tmp[param] = data

    day_data.append(tmp)

# Concatenate all days for each parameter
tmp = {}
for param in obs_params:
    all_days = []
    for dd in day_data:
        all_days.append(dd[param])
    concat = pd.concat(all_days, axis=0)
    tmp[param] = concat


# Create dictionary with each parameter as a key and pandas df as value
obs = {}
dts = tmp[list(obs_params.keys())[0]].index.values.astype('M8[s]').astype('O')
#obs[G.PAR_NAMES] = obs_params
stations = {}
for stat_key in inp_station_names: 
    if np.any(stations_meta_use['ABBR'] == stat_key):
        #stations[stat_key] = {}
        #stations[stat_key][G.STAT_META] = stations_meta_use[stations_meta_use['ABBR'] == stat_key]
        values = np.zeros((len(dts),len(obs_params)))
        param_names = []
        for i,param in enumerate(obs_params):
            values[:,i] = tmp[param][stat_key]
            param_names.append(obs_params[param])
            #stations[stat_key][G.PAR][param] = tmp[param][stat_key]
        df = pd.DataFrame(values, index=dts, columns=param_names)
        stations[stat_key] = df

    else:
        #print(stat_key)
        pass


obs[G.STAT] = stations
data = {}
data[G.OBS] = obs
data[G.HIST] = [hist_tag]

data[G.STAT_META] = {}
for stat_key in inp_station_names: 
    if np.any(stations_meta_use['ABBR'] == stat_key):
        data[G.STAT_META][stat_key] = stations_meta_use[stations_meta_use['ABBR'] == stat_key]

# Save
pickle.dump(data, open(CN.obs_path, 'wb'))

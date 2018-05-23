import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

############ USER INPUT #############
obs_path = '../obs_out/'
obs_inp_file = '20180103sfc.'
# obs case name (name of obs pkl file in data folder)
obs_case_name = 'burglind'
out_pickle_file_path = '../data/obs_'+obs_case_name+'.pkl'
MISSING_VALUE = -9999
sample_rate = '1H'
OBS = 'obs'
### use_params
# VMAX_10M1: hourly max gust
# FF_10M   : 10min mean wind speed @10m
# DD_10M   : 10min mean wind direction @10m
use_params = ['VMAX_10M1', 'FF_10M', 'DD_10M']
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


# Create dictionary with each parameter as a key and pandas df as value
out = {}
obs = {}
for param in use_params:
    mask_inds = np.argwhere(inp_params == param).squeeze()
    values = inp_values[mask_inds]
    dts = [inp_dts[i] for i in mask_inds]
    # create data frame
    data = pd.DataFrame(values, index=dts, columns=inp_station_names)
    data = data.resample(sample_rate).mean()
    obs[param] = data
out[OBS] = obs

# Save
pickle.dump(out, open(out_pickle_file_path, 'wb'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

############ USER INPUT #############
# obs case name (name of obs pkl file in data folder)
obs_case_name = 'burglind'
# model case name (name of folder with model data in 'mod_path'
model_case_name = 'burglind_ref'
# time window of model run
# THIS HAS TO BE ADJUSTED IF THE MODEL TIME WINDOW IS CHANGED!
# OBSERVATIONS ARE ASSUMED TO BE AVAILABLE FOR THIS TIME WINDOW!
start_time = datetime(2018,1,3,0,0)
end_time = datetime(2018,1,4,0,0)
#end_time = datetime(2018,1,3,0,0)
# path to model fortran files
mod_path = '../model_out/'+model_case_name+'/'
# path to pickle file of obs created by python script 01_prep_obs.py
obs_pickle_file_path = '../data/OBS_'+obs_case_name+'.pkl'
# file name of pickle file containing all the data
data_pickle_file_path = '../data/OBS_'+obs_case_name+'_MODEL_'+model_case_name+'.pkl'
# fort.700 does not contain data but station information!
mod_stations_file = '../model_out/'+model_case_name+'/fort.700' 
# starting index of fortran files
ind0 = 701
# header of fortran output files
file_headers = ['ntstep','k_bra','tcm','zvp10','zvp30','zv_bra','zvpb',
                'zuke','zvke','zukem1','zvkem1','zukem2','zvkem2']
MODEL = 'model'
OBS = 'obs'
#####################################


# read model station names
mod_stations = np.genfromtxt(mod_stations_file, skip_header=2, dtype=np.str)[:,0]
#mod_stations = ['ABO','AEG'] # debug

# load main data file
data = pd.read_pickle(obs_pickle_file_path)
use_params = list(data[OBS].keys())
# add entry for model data
data[MODEL] = {}

# read model data
for i,_ in enumerate(mod_stations):
    ind = i+ind0 
    # construct file path
    mod_file_path = mod_path + 'fort.' + str(ind)
    #print(mod_file_path)
    stat_key = mod_stations[i]
    # try to read file. some files are badly written by fortran (e.g. one row only contains 11 instead of 12 cols)
    try:
        values = np.loadtxt(mod_file_path, delimiter=',')
        print(stat_key)
        data[MODEL][stat_key] = {}
        # add to data dictionary
        for j,fh in enumerate(file_headers):
            data[MODEL][stat_key][fh] = values[:,j]
    except:
        print('ERROR LOADING STATION: ',stat_key)
print('##################')


sel_obs = data[OBS][use_params[0]][start_time:end_time]
# save observed hourly values
delete_keys = []
for key in data[MODEL].keys():
    if key in sel_obs.columns:
        #data[MODEL][key]['obs'] = sel_obs[key].values
        pass
    else:
        delete_keys.append(key)

print('delete keys that are not in observations:')
for key in delete_keys:
    print(key)
    del data[MODEL][key]
# save datetime (convert to suitable dt64 object first, then to normal python datetime object)
data['dts'] = sel_obs.index.values.astype('M8[s]').astype('O')

# final structured output
out = {}
out[MODEL] = data[MODEL]
out['obs_dts'] = data['dts']
out['model_dts'] = pd.date_range(start_time, end_time, freq='10s').values.astype('M8[s]').astype('O')
out['model_file_headers'] = file_headers
out['station_names'] = list(data[MODEL].keys())
out[OBS] = {}
for key in out[MODEL].keys():
    out[OBS][key] = {}
    for param in use_params:
        out[OBS][key][param] = data[OBS][param][key].values
        

# save output file
pickle.dump(out, open(data_pickle_file_path, 'wb'))

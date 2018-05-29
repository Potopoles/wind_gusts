import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import globals as G

############ USER INPUT #############
# obs case name (name of obs pkl file in data folder)
obs_case_name = 'burglind'
# model case name (name of folder with model data in 'mod_path'
model_case_name = 'burglind_ref'
# obs model combination case name
obs_model_case_name = 'OBS_'+obs_case_name+'_MODEL_'+model_case_name
# time window of model run
# THIS HAS TO BE ADJUSTED IF THE G.MODEL TIME WINDOW IS CHANGED!
# G.OBSERVATIONS ARE ASSUMED TO BE AVAILABLE FOR THIS TIME WINDOW!
start_time = datetime(2018,1,3,0,0)
#end_time = datetime(2018,1,4,0,0)
# time step [s] of model
model_dt = 10
# path to model fortran files
mod_path = '../model_out/'+model_case_name+'/'
# path to pickle file of obs created by python script 01_prep_obs.py
obs_pickle_file_path = '../data/OBS_'+obs_case_name+'.pkl'
# file name of pickle file containing all the data
data_pickle_file_path = '../data/'+obs_model_case_name+'.pkl'
# fort.700 does not contain data but station information!
mod_stations_file = '../model_out/'+model_case_name+'/fort.700' 
# starting index of fortran files
ind0 = 701
# header of fortran output files
model_params = ['ntstep','k_bra','tcm','zvp10','zvp30','zv_bra','zvpb',
                'zuke','zvke','zukem1','zvkem1','zukem2','zvkem2']
#####################################


# read model station names
mod_stations = np.genfromtxt(mod_stations_file, skip_header=2, dtype=np.str)[:,0]
#mod_stations = ['ABO','AEG'] # debug

# load main data file
data = pd.read_pickle(obs_pickle_file_path)
# stations to use as given by observation data set
use_stations = list(data[G.OBS][G.STAT].keys())
# add entry for model data
data[G.MODEL] = {G.STAT:{}}
data[G.MODEL][G.PAR_NAMES] = model_params

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
        if stat_key in use_stations:
            print('use ' + stat_key)
            data[G.MODEL][G.STAT][stat_key] = {G.PAR:{}}
            # add to data dictionary
            for j,mp in enumerate(model_params):
                data[G.MODEL][G.STAT][stat_key][G.PAR][mp] = values[:,j]
        else:
            print('do not use ' + stat_key)
    except:
        print('ERROR LOADING STATION: ',stat_key)
print('##################')

# calculate datetime of each time step
dts = [start_time + timedelta(seconds=nsec) for nsec in values[:,0]*model_dt]
data[G.MODEL][G.DTS] = dts

# save names of used stations
data[G.STAT_NAMES] = list(data[G.MODEL][G.STAT].keys())

# save output file
pickle.dump(data, open(data_pickle_file_path, 'wb'))

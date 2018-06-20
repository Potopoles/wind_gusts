#import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from datetime import datetime
#import copy
import pickle
#from sklearn.linear_model import LinearRegression
#from functions import calc_model_fields, join_model_and_obs, \
#                        join_model_runs, join_all_stations
import globals as G
#from filter import EntryFilter
from namelist_cases import Case_Namelist

############ USER INPUT #############
case_index = 0
CN = Case_Namelist(case_index)
#min_gust = 10
#####################################

def plot_error(obs, gust):
    plt.scatter(obs, gust-obs)
    ax = plt.gca()
    ax.axhline(y=0,c='k')
    ax.set_xlim(0,20)
    ax.set_ylim(-20,20)
    plt.show()



# load data
data = pickle.load( open(CN.mod_path, 'rb') )


stat_key = 'AEG'
stat_keys = ['AEG','ABO', 'AIG']
lm_run = '2018010300'

#print(data[G.MODEL][G.STAT][stat_key][G.RAW][lm_run])
#quit()
#raw_data = data[G.MODEL][G.STAT][stat_key][G.RAW][lm_run][['tcm','zvp10']] 



tcm_all = []
zvp10_all = []
obs_all = []

for stat_key in stat_keys:
    tcm = data[G.MODEL][G.STAT][stat_key][G.RAW][lm_run]['tcm']
    tcm[tcm < 5E-4] = 5E-4
    tcm = np.sqrt(tcm)
    tcm_all.append(tcm.rename(stat_key))

    zvp10 = data[G.MODEL][G.STAT][stat_key][G.RAW][lm_run]['zvp10']
    zvp10_all.append(zvp10.rename(stat_key))

    obs = data[G.OBS][G.STAT][stat_key][G.OBS_GUST_SPEED]
    model_time_tmp = tcm.resample('H').max()
    obs = pd.concat([obs,model_time_tmp], axis=1, join='inner')[G.OBS_GUST_SPEED]
    obs_all.append(obs.rename(stat_key))
    


tcm = pd.concat(tcm_all, axis=1)
zvp10 = pd.concat(zvp10_all, axis=1)
obs = pd.concat(obs_all, axis=1)

obs_flat = obs.values.flatten()


N = obs_flat.shape[0]
alpha1 = 5
dalpha1 = np.Inf
learning_rate = 1

while np.abs(dalpha1) > 0.1:
    # calc current time step gusts
    gust = zvp10 + alpha1*tcm*zvp10

    # find maximum hourly gust
    tcm_max = np.full(obs.shape, np.nan)
    zvp10_max = np.full(obs.shape, np.nan)
    gust_max = np.full(obs.shape, np.nan)
    for i,stat_key in enumerate(stat_keys):
        maxid = gust[stat_key].resample('H').agg(np.argmax)
        tcm_max[:,i] = tcm[stat_key][maxid].values
        zvp10_max[:,i] = zvp10[stat_key][maxid].values
        gust_max[:,i] = gust[stat_key][maxid].values

    tcm_max = tcm_max.flatten()
    zvp10_max = zvp10_max.flatten()
    gust_max = gust_max.flatten()
    error = np.sqrt(np.sum((obs_flat - gust_max)**2)/N)

    # gradient of parameters
    dalpha1 = -2/N * np.sum( tcm_max*zvp10_max * (obs_flat - gust_max) )
    #print('dalpha ' + str(dalpha1))

    alpha1 = alpha1 - learning_rate * dalpha1
    #print('alpha ' + str(alpha1))

print('alpha ' + str(alpha1))
plot_error(obs_flat, gust_max)
#quit()
quit()



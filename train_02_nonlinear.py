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




# load data
data = pickle.load( open(CN.mod_path, 'rb') )


stat_key = 'AEG'
lm_run = '2018010300'

#print(data[G.MODEL][G.STAT][stat_key][G.RAW][lm_run])
#quit()
#raw_data = data[G.MODEL][G.STAT][stat_key][G.RAW][lm_run][['tcm','zvp10']] 
tcm = data[G.MODEL][G.STAT][stat_key][G.RAW][lm_run]['tcm']
tcm[tcm < 5E-4] = 5E-4
tcm = np.sqrt(tcm)
zvp10 = data[G.MODEL][G.STAT][stat_key][G.RAW][lm_run]['zvp10']


# load and cut observations
obs = data[G.OBS][G.STAT][stat_key][G.OBS_GUST_SPEED]
model_time_tmp = tcm.resample('H').max()
obs = pd.concat([obs,model_time_tmp], axis=1, join='inner')[G.OBS_GUST_SPEED].values


N = obs.shape[0]
alpha1 = 5
alpha2 = 0.09
dalpha1 = np.Inf
dalpha2 = np.Inf
learning_rate = 0.01

while max(np.abs(dalpha1), np.abs(dalpha2)) > 0.1:
    # calc current time step gusts
    gust = zvp10 + alpha1*tcm*zvp10 + alpha2*tcm*zvp10**2

    # find maximum hourly gust
    maxid = gust.resample('H').agg(np.argmax)
    tcm_max = tcm[maxid].values
    zvp10_max = zvp10[maxid].values
    gust_max = gust[maxid].values

    error = np.sum((obs - gust_max)**2)
    print(error)

    # gradient of parameters
    dalpha1 = -2/N * np.sum( tcm_max*zvp10_max * (obs - gust_max) )
    dalpha2 = -2/N * np.sum( tcm_max*zvp10_max**2 * (obs - gust_max) )

    alpha1 = alpha1 - learning_rate * dalpha1
    alpha2 = alpha2 - learning_rate * dalpha2

print('alpha1 ' + str(alpha1))
print('alpha2 ' + str(alpha2))
plt.scatter(obs, gust_max-obs)
ax = plt.gca()
ax.axhline(y=0,c='k')
ax.set_xlim(0,20)
ax.set_ylim(-20,20)
plt.show()
quit()



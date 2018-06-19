#import os
import numpy as np
import matplotlib.pyplot as plt
#from datetime import datetime
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
## do not plot (0) show plot (1) save plot (2)
#i_plot = 1
i_stat = 'ABO'
#####################################



# load data
data = pickle.load( open(CN.mod_path, 'rb') )

#model = data[G.MODEL][G.STAT][i_stat][G.RAW]['2018010300']
model = data[G.MODEL][G.STAT][i_stat][G.RAW]['2018061000']
print(model.columns)

plt.plot(model.qvflx)
#plt.plot(model.qvl1)
#plt.plot(model.shflx)
#plt.plot(model.Tl1)
plt.show()

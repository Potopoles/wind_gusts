import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle


############ USER INPUT #############
# i_method = 1: estimate from zvp10 and ustar
# i_method = 2: estimate from zvp30 and ustar
# i_method = 3: brasseur
# i_method = 4: estimate from zvp10 and ustar and gust factor
i_method = 4
data_pickle_path = '../data/model_and_obs.pkl'
file_headers = ['ntstep','k_bra','tcm','zvp10','zvp30','zv_bra','zvpb',
                'zuke','zvke','zukem1','zvkem1','zukem2','zvkem2']
MODEL = 'model'
OBS = 'obs'
#####################################

data = pickle.load( open(data_pickle_path, 'rb') )

station_names = data['station_names']
nstat = len(station_names)

# Prepare index mask to map model output to observation values
nhrs = len(data['obs_dts'])
hr_inds = np.zeros((nhrs,360))
for i in range(0,nhrs):
    hr_inds[i,:] = i*360 + np.arange(0,360)
hr_inds = hr_inds.astype(np.int)



# error fields
mod = np.full((nhrs,nstat),np.nan)
obs = np.full((nhrs,nstat),np.nan)
mod_err = np.full((nhrs,nstat),np.nan)
abs_err = np.full((nhrs,nstat),np.nan)


# loop through all stations
for si,stat in enumerate(station_names):
    #print(stat)

    if i_method == 1: 

        tcm = data[MODEL][stat]['tcm']
        zcm = tcm
        zcm[zcm < 5E-4] = 5E-4
        zsqcm = np.sqrt(zcm)
        zvp10 = data[MODEL][stat]['zvp10']
        gust = zvp10 + 3.0 * 2.4 * zsqcm * zvp10

    elif i_method == 2:

        tcm = data[MODEL][stat]['tcm']
        zcm = tcm
        zcm[zcm < 5E-4] = 5E-4
        zsqcm = np.sqrt(zcm)
        zvp30 = data[MODEL][stat]['zvp30']
        zvpb = data[MODEL][stat]['zvpb']
        gust = zvp30 + 3.0 * 2.4 * zsqcm * zvpb

    elif i_method == 3:

        gust = data[MODEL][stat]['zv_bra']

    elif i_method == 4:

        tcm = data[MODEL][stat]['tcm']
        zcm = tcm
        zcm[zcm < 5E-4] = 5E-4
        zsqcm = np.sqrt(zcm)
        zvp10 = data[MODEL][stat]['zvp10']
        gust = zvp10 + (3.0 * 2.4 + 0.09 * zvp10) * zsqcm * zvp10
    
    for hr in range(0,nhrs):
        inds = hr_inds[hr]
        hr_max_gust = np.max(gust[inds])
        #print(hr_max_gust, data[MODEL][stat]['obs'][hr])

        mod[hr,si] = hr_max_gust
        obs[hr,si] = data[OBS][stat]['VMAX_10M1'][hr] 

mod_err = mod - obs
abs_err = np.abs(mod - obs)


print('ERRORS')
print(np.nanmean(mod_err))
print(np.nanmean(abs_err))

mod_time_mean = np.nanmean(mod, axis=0)
mod_stat_mean = np.nanmean(mod, axis=1)
obs_time_mean = np.nanmean(obs, axis=0)
obs_stat_mean = np.nanmean(obs, axis=1)
line1, = plt.plot(mod_stat_mean)
line2, = plt.plot(obs_stat_mean)
plt.legend([line1,line2],labels=['model','obs'])
plt.show()

    


    


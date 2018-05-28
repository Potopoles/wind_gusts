import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from sklearn import metrics
from sklearn.linear_model import LinearRegression


############ USER INPUT #############
# obs case name (name of obs pkl file in data folder)
obs_case_name = 'burglind'
# model case name (name of folder with model data in 'mod_path'
model_case_name = 'burglind_ref'
# obs model combination case name
obs_model_case_name = 'OBS_'+obs_case_name+'_MODEL_'+model_case_name
# mode of evaluation: either ALL_STATIONS (1 plot for each station) or MEAN_OVER_STATIONS
#eval_mode = 'ALL_STATIONS'
#eval_mode = 'MEAN_OVER_STATIONS'
eval_mode = 'ABO'
# should a plot be created?
i_create_plot = 0
# do not plot (0) show plot (1) save plot (2)
i_plot = 2
# path of input obs_model pickle file
data_pickle_path = '../data/'+obs_model_case_name+'.pkl'
plot_base_dir = '../plots/'
plot_case_dir = plot_base_dir + obs_model_case_name + '/'
MODEL = 'model'
OBS = 'obs'
STAT = 'stations'
PAR = 'params'
#####################################

# create directories
if i_plot > 1 and not os.path.exists(plot_case_dir):
    os.mkdir(plot_case_dir)

# load data
data = pickle.load( open(data_pickle_path, 'rb') )

station_names = np.asarray(data['station_names'])
nstat = len(station_names)

# Prepare index mask to map model output to observation values
nhrs = len(data[OBS]['dts'])
hr_inds = np.zeros((nhrs,360))
for i in range(0,nhrs):
    hr_inds[i,:] = i*360 + np.arange(0,360)
hr_inds = hr_inds.astype(np.int)


# value fields
mod_gust = np.full((nhrs,nstat),np.nan)
obs_gust = np.full((nhrs,nstat),np.nan)
k_max = np.full((nhrs,nstat),np.nan)
#mod_wind = np.full((nhrs,nstat),np.nan)
#obs_wind = np.full((nhrs,nstat),np.nan)


# loop through all stations
for si,stat in enumerate(station_names):

    for hr in range(0,nhrs):
        inds = hr_inds[hr]
        obs_gust[hr,si] = data[OBS][STAT][stat][PAR]['VMAX_10M1'][hr] 
        #mod_wind[hr,si] = np.mean(data[MODEL][stat]['zvp10'][inds])
        #obs_wind[hr,si] = data[OBS][stat]['FF_10M'][hr] 


        gust = data[MODEL][STAT][stat][PAR]['zv_bra']
        k_bra = data[MODEL][STAT][stat][PAR]['k_bra']

        # calc and save model max gust
        for hr in range(0,nhrs):
            inds = hr_inds[hr]
            # find maximum gust
            hr_max_gust = np.max(gust[inds])
            max_ind = np.argmax(gust[inds])
            k_bra_max = k_bra[inds][max_ind]

            mod_gust[hr,si] = hr_max_gust
            k_max[hr,si] = k_bra_max


error = mod_gust - obs_gust


title = '500 stations'
si = np.argwhere(station_names == stat).squeeze()
si = np.arange(0,500)

if i_plot > 0:
    # regression
    reg = LinearRegression(fit_intercept=True)

    # plot preparation
    fig,axes = plt.subplots(1,1,figsize=(5,4))

    ax = axes

    ## prepare feature matrix
    X = error[:,si].flatten().reshape(-1,1)
    y = k_max[:,si].flatten()
    ##print(X.shape)
    ##print(y.shape)

    # remove nans
    mask = np.isnan(X[:,0])
    #mask[np.isnan(y)] = True
    X = X[~mask]
    y = y[~mask]

    # determine max/min y
    #ymax = max(np.max(y),ymax)
    #ymin = min(np.min(y),ymin)

    # fit regression and draw line
    reg.fit(X,y)
    line = reg.predict(X)

    # plotting
    ax.scatter(error[:,si], k_max[:,si], color='black', marker=".")
    ax.plot(X, line, color='red')
    ax.set_xlabel('error (MOD-OBS) [m/s]')
    ax.set_ylabel('model level of max gust')
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_title('k max vs error')
    # set axes limits in each ax
    #ax.set_ylim((ymin,ymax))

    # finish plot
    #plt.suptitle(title)

    if i_plot == 1:
        plt.show()
    elif i_plot > 1:
        plot_name = plot_case_dir + 'kmax_vs_error.png'
        plt.savefig(plot_name)
        plt.close('all')



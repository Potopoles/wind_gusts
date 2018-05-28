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
i_plot = 1
# gust methods to calculate and plot
# i_method = 1: estimate from zvp10 and ustar
# i_method = 2: estimate from zvp30 and ustar
# i_method = 3: brasseur
# i_method = 4: estimate from zvp10 and ustar and gust factor
i_methods = [1,3,4]
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
nmethods = len(i_methods)

# Prepare index mask to map model output to observation values
nhrs = len(data[OBS]['dts'])
hr_inds = np.zeros((nhrs,360))
for i in range(0,nhrs):
    hr_inds[i,:] = i*360 + np.arange(0,360)
hr_inds = hr_inds.astype(np.int)


# value fields
mod_gust = np.full((nhrs,nstat,4),np.nan)
obs_gust = np.full((nhrs,nstat),np.nan)
#mod_wind = np.full((nhrs,nstat),np.nan)
#obs_wind = np.full((nhrs,nstat),np.nan)


# loop through all stations
for si,stat in enumerate(station_names):

    for hr in range(0,nhrs):
        inds = hr_inds[hr]
        obs_gust[hr,si] = data[OBS][STAT][stat][PAR]['VMAX_10M1'][hr] 
        #mod_wind[hr,si] = np.mean(data[MODEL][stat]['zvp10'][inds])
        #obs_wind[hr,si] = data[OBS][stat]['FF_10M'][hr] 

    for mi in i_methods:


        if mi == 1:

            tcm = data[MODEL][STAT][stat][PAR]['tcm']
            zcm = tcm
            zcm[zcm < 5E-4] = 5E-4
            zsqcm = np.sqrt(zcm)
            zvp10 = data[MODEL][STAT][stat][PAR]['zvp10']
            gust = zvp10 + 3.0 * 2.4 * zsqcm * zvp10

        if mi == 2:

            tcm = data[MODEL][STAT][stat][PAR]['tcm']
            zcm = tcm
            zcm[zcm < 5E-4] = 5E-4
            zsqcm = np.sqrt(zcm)
            zvp30 = data[MODEL][STAT][stat][PAR]['zvp30']
            zvpb = data[MODEL][STAT][stat][PAR]['zvpb']
            gust = zvp30 + 3.0 * 2.4 * zsqcm * zvpb

        if mi == 3:

            gust = data[MODEL][STAT][stat][PAR]['zv_bra']

        if mi == 4:

            tcm = data[MODEL][STAT][stat][PAR]['tcm']
            zcm = tcm
            zcm[zcm < 5E-4] = 5E-4
            zsqcm = np.sqrt(zcm)
            zvp10 = data[MODEL][STAT][stat][PAR]['zvp10']
            gust = zvp10 + (3.0 * 2.4 + 0.09 * zvp10) * zsqcm * zvp10
    
        # calc and save model max gust
        for hr in range(0,nhrs):
            inds = hr_inds[hr]
            # find maximum gust
            hr_max_gust = np.max(gust[inds])

            mod_gust[hr,si,mi-1] = hr_max_gust


# calculate scores
# error fields
mod_err_gust = np.full((nhrs,nstat,4),np.nan)
abs_err_gust = np.full((nhrs,nstat,4),np.nan)
for mi in i_methods:
    mod_err_gust[:,:,mi-1] = mod_gust[:,:,mi-1] - obs_gust
abs_err_gust = np.abs(mod_err_gust)

#stat = 'SAE'
title = '500 stations'
si = np.argwhere(station_names == stat).squeeze()
si = np.arange(0,500)
#si = 1
#print(si)

###################### PLOT ABSOLUTE ERROR
if i_plot > 0:
    # regression
    reg = LinearRegression(fit_intercept=True)

    # plot preparation
    fig,axes = plt.subplots(1,3,figsize=(14,4))
    ymax = -np.Inf
    ymin = np.Inf

    # loop over axes and gust calc method
    for i,mi in enumerate(i_methods):
        ax = axes[i]

        # prepare feature matrix
        X = obs_gust[:,si].flatten().reshape(-1,1)
        y = mod_err_gust[:,si,mi-1].flatten()
        #print(X.shape)
        #print(y.shape)

        # remove nans
        mask = np.isnan(X[:,0])
        #mask[np.isnan(y)] = True
        X = X[~mask]
        y = y[~mask]

        # determine max/min y
        ymax = max(np.max(y),ymax)
        ymin = min(np.min(y),ymin)

        # fit regression and draw line
        reg.fit(X,y)
        line = reg.predict(X)

        # plotting
        ax.scatter(obs_gust[:,si], mod_err_gust[:,si,mi-1], color='black', marker=".")
        ax.plot(X, line, color='red')
        ax.set_xlabel('Observed gust (OBS) [m/s]')
        if i == 0:
            ax.set_ylabel('Model absolute error (MOD-OBS) [m/s]')
        ax.axhline(0, color='k', linewidth=0.8)
        ax.set_title('Method '+str(mi))
    # set axes limits in each ax
    for ax in axes:
        ax.set_ylim((ymin,ymax))

    # finish plot
    plt.suptitle(title)

    if i_plot == 1:
        plt.show()
    elif i_plot > 1:
        plot_name = plot_case_dir + 'scatter.png'
        plt.savefig(plot_name)
        plt.close('all')


###################### CALCULATE ERROR MEASURES
# CONTINUOUS ERROR MEASURES
error_measures = {}
error_measures['mean_abs_err'] = np.full((nstat,nmethods), np.nan)
error_measures['r2'] = np.full((nstat,nmethods), np.nan)
error_measures['explained_var'] = np.full((nstat,nmethods), np.nan)
#sis = np.arange(0,600)
for si in range(0,nstat):
    y_obs = obs_gust[:,si]

    # find nans
    mask = np.isnan(y_obs)
    y_obs = y_obs[~mask]

    for i,mi in enumerate(i_methods):
        y_mod = mod_gust[:,si,mi-1].flatten()
        y_mod = y_mod[~mask]
        error_measures['mean_abs_err'][si,i] = metrics.mean_absolute_error(y_obs, y_mod)
        error_measures['r2'][si,i] = metrics.r2_score(y_obs, y_mod)
        error_measures['explained_var'][si,i] = metrics.explained_variance_score(y_obs, y_mod)


# CATEGORICAL ERROR MEASURES


# print measures to file
file_name = plot_case_dir + 'scores.txt'
with open(file_name, 'w') as f:
    f.write('i_methods ' + str(i_methods) + '\n')
    for key,meas in error_measures.items():
        mean_meas = np.mean(meas, axis=0)
        text = 'station mean ' + key + ' ' + str(mean_meas)
        print(text)
        f.write(text + '\n')


###################### PLOT CONTINUOUS ERROR MEASURES
if i_plot > 0:
    fig,axes = plt.subplots(1,3,figsize=(14,4))
    i = 0
    for key,meas in error_measures.items():
        ax = axes[i]
        if key == 'mean_abs_err':
            bins = np.arange(0,20.1,1)
        else:
            bins = np.arange(-1,1.1,0.1)
        ax.hist(meas,bins=bins, histtype='step',
                color=['red','orange','black'], label=['Method 1', 'Method 2', 'Method 4'])
        ax.axvline(0,color='k')
        if key == 'mean_abs_err':
            ax.legend(loc='upper right')
            ax.set_xlabel('Absolute error [m/s]')
        else:
            ax.legend(loc='upper left')
            ax.set_xlabel(']-inf,1]')
        ax.set_title(key)
        i += 1

    if i_plot == 1:
        plt.show()
    elif i_plot > 1:
        plot_name = plot_case_dir + 'err_continuous.png'
        plt.savefig(plot_name)
        plt.close('all')


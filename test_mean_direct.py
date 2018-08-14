import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
from sklearn.linear_model import LinearRegression
from functions import calc_model_fields, join_model_and_obs, \
                        join_model_runs, join_all_stations
import globals as G
from filter import EntryFilter
from namelist_cases import Case_Namelist

############ USER INPUT #############
case_index = 15
#case_index = 0
CN = Case_Namelist(case_index)
# do not plot (0) show plot (1) save plot (2)
i_plot = 1
# model fields to calculate 
i_model_fields = [G.MODEL_MEAN_WIND]
#min_gust_levels = [0,5,10,20]
min_gust_levels = [0]
label = ''
#####################################

# create directories
if i_plot > 1 and not os.path.exists(CN.plot_path):
    os.mkdir(CN.plot_path)

EF = EntryFilter()

for min_gust in min_gust_levels:
    print('########## ' + str(min_gust) + ' ##########')

    # load data
    print('load')
    data = pickle.load( open(CN.mod_path, 'rb') )
    stat_names = list(data[G.MODEL][G.STAT].keys())
    lm_runs = list(data[G.MODEL][G.STAT][stat_names[0]][G.RAW].keys())
    model_mean = np.zeros( (len(lm_runs)*12, len(stat_names)) )
    obs_mean   = np.zeros( (len(lm_runs)*12, len(stat_names)) )
    dates = []

    #print(data[G.OBS][G.STAT][stat_names[0]]['obs_mean_wind'])
    #quit()

    for lmi,lm_run in enumerate(lm_runs):
        #lm_run = '2017121412'
        #lmi = 0

        dts = data[G.MODEL][G.STAT][stat_names[0]][G.RAW][lm_run].resample('H',\
                                loffset=timedelta(hours=1)).mean().index[12:24]
        dates.extend(dts)
        #print(dates)
        #quit()
        for si,stat_name in enumerate(stat_names):
            #stat_name = stat_names[0]
            #si = 0
            lminds = np.arange(lmi*12,(lmi+1)*12).astype(np.int)
            #print(data[G.MODEL][G.STAT][stat_name][G.RAW][lm_run].resample('H',\
            #                        loffset=timedelta(hours=1)).mean()['zvp10'])
            #quit()
            model_mean[lminds,si] = data[G.MODEL][G.STAT][stat_name][G.RAW][lm_run].resample('H',\
                                    loffset=timedelta(hours=1)).mean()['zvp10'].values[12:24]

            obs_mean[lminds,si] = data[G.OBS][G.STAT][stat_name]['obs_mean_wind'].loc[dts].values
            #print(data[G.OBS][G.STAT][stat_name]['obs_mean_wind'].loc[dts])
        #quit()

    ##################### MEAN PLOT
    #model_mean = np.mean(model_mean, axis=1)
    #obs_mean = np.nanmean(obs_mean, axis=1)
    #fig, axes = plt.subplots(2,1, figsize=(10,5))
    #ax = axes[1]
    #line1, = ax.plot(dates, model_mean, color='red', label='model')
    #line2, = ax.plot(dates, obs_mean, color='black', label='obs')
    #ax.legend([line1, line2], ['model', 'obs'])
    #ax.set_xlabel('date')
    #ax.set_ylabel('wind [m/s]')
    #ax.set_ylim((0,10))
    #ax = axes[0]
    #line1, = ax.plot(dates, model_mean - obs_mean, color='black', label='model error')
    #ax.axhline(y=0)
    #ax.legend([line1], ['model error'])
    #ax.set_xlabel('date')
    #ax.set_ylabel('wind error [m/s]')
    #ax.set_ylim((-3,3))
    #plt.show()
    #quit()
    #####################


    ##################### HISTOGRAM
    obs_mean = obs_mean.flatten()
    obs_mask = np.isnan(obs_mean)
    obs_mean = obs_mean[~obs_mask]
    model_mean = model_mean.flatten()[~obs_mask]
    xbins = np.arange(0,25,1)
    diffbins = np.arange(-20,20,2)
    fig, axes = plt.subplots(1,2, figsize=(12,5))
    ax = axes[0]
    h1 = ax.hist(model_mean, bins=xbins, color=(1.0,0.,0.,0.4), label='MOD', log=True)
    ax.set_xlabel('wind [m/s]')
    ax.set_title('MODEL (red) and OBS (grey)')
    h2 = ax.hist(obs_mean, bins=xbins, color=(0.1,0.1,0.1,0.4), label='OBS', log=True)
    #ax.legend([h1, h2], ['MOD', 'OBS'])
    ax = axes[1]
    ax.hist((model_mean - obs_mean), bins=diffbins, log=True)
    ax.set_xlabel('wind difference [m/s]')
    ax.set_title('MODEL - OBS')
    plt.show()
    quit()
    line2, = ax.plot(dates, obs_mean, color='black', label='obs')
    ax.legend([line1, line2], ['model', 'obs'])
    ax.set_ylim((0,10))
    ax = axes[0]
    line1, = ax.plot(dates, model_mean - obs_mean, color='black', label='model error')
    ax.axhline(y=0)
    ax.legend([line1], ['model error'])
    ax.set_xlabel('date')
    ax.set_ylabel('wind error [m/s]')
    ax.set_ylim((-3,3))
    plt.show()
    quit()
    #####################


    ##################### SPAGHETTI PLOT
    fig, axes = plt.subplots(1,2, figsize=(18,5))
    for i in range(30):
        ax = axes[1]
        line1, = ax.plot(dates, model_mean[:,i], color='red', label='model')
        line2, = ax.plot(dates, -obs_mean[:,i], color='black', label='obs')
        if i == 0:
            ax.legend([line1, line2], ['model', 'obs'])
        ax = axes[0]
        line1, = ax.plot(dates, model_mean[:,i] - obs_mean[:,i], color='black', label='model error')
        if i == 0:
            ax.legend([line1], ['model error'])

    model_mean_med = np.median(model_mean, axis=1)
    obs_mean_med = np.nanmedian(obs_mean, axis=1)
    diff_mean_med = np.nanmedian(model_mean - obs_mean, axis=1)

    ax = axes[1]
    ax.plot(dates, model_mean_med, color='orange', lineWidth=3)
    ax.plot(dates, -obs_mean_med, color='orange', lineWidth=3)
    ax.axhline(y=0)
    ax.set_xlabel('date')
    ax.set_ylabel('wind [m/s]')
    ax.set_ylim((-30,30))
    ax = axes[0]
    ax.plot(dates, -diff_mean_med, color='orange', lineWidth=3)
    ax.axhline(y=0)
    ax.set_xlabel('date')
    ax.set_ylabel('wind error [m/s]')
    ax.set_ylim((-20,20))
    plt.show()
    #####################

    ##################### ERROR SCATTER
    #err_mean = model_mean - obs_mean
    #ax = plt.gca()
    #ax.scatter(obs_mean, err_mean)
    #ax.axhline(y=0, color='k')
    #plt.show()
    #####################

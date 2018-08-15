import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from sklearn.linear_model import LinearRegression
from functions import calc_model_fields, join_model_and_obs, \
                        join_model_runs, join_all_stations
import globals as G
from filter import EntryFilter
from namelist_cases import Case_Namelist

############ USER INPUT #############
case_index = 15
#case_index = 10
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

    #print(data[G.OBS][G.STAT].keys())
    #quit()

    # calculate gusts
    print('calc gust')
    data = calc_model_fields(data, i_model_fields)
    # join model and obs
    print('join model obs')
    data = join_model_and_obs(data)
    ## filter according to min gust strength
    #print('filter according gust')
    #data = EF.filter_according_obs_gust(data, min_gust)
    # join all model runs
    print('join runs')
    data = join_model_runs(data)
    stations = list(data[G.BOTH][G.STAT].keys())
    nvals = int(len(data[G.BOTH][G.STAT][stations[0]][G.MODEL_MEAN_WIND].values)/2)
    sel_inds = [np.arange(i,i+12) for i in range(12,nvals*2,24)]
    sel_inds = np.asarray([item for sublist in sel_inds for item in sublist])
    #print(sel_inds)


    model_mean = np.zeros( (nvals, len(stations)) )
    obs_mean = np.zeros( (nvals, len(stations)) )
    for i,stat in enumerate(stations):
        #stat = 'ABO'
        mm = data[G.BOTH][G.STAT][stat][G.MODEL_MEAN_WIND]
        om = data[G.BOTH][G.STAT][stat][G.OBS_MEAN_WIND]
        #print(om.values)
        model_mean[:,i] = mm.values[sel_inds]
        obs_mean[:,i] = om.values[sel_inds]
    dates = mm.index[sel_inds]

    #print(dates)
    #quit()

    ##################### MEAN PLOT
    model_mean = np.mean(model_mean, axis=1)
    obs_mean = np.nanmean(obs_mean, axis=1)
    fig, axes = plt.subplots(2,1, figsize=(10,5))
    ax = axes[1]
    line1, = ax.plot(dates, model_mean, color='red', label='model')
    line2, = ax.plot(dates, obs_mean, color='black', label='obs')
    ax.legend([line1, line2], ['model', 'obs'])
    ax.set_xlabel('date')
    ax.set_ylabel('wind [m/s]')
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


    ##################### HISTOGRAM
    #obs_mean = obs_mean.flatten()
    #obs_mask = np.isnan(obs_mean)
    #obs_mean = obs_mean[~obs_mask]
    #model_mean = model_mean.flatten()[~obs_mask]
    #xbins = np.arange(0,25,1)
    #diffbins = np.arange(-20,20,2)
    #fig, axes = plt.subplots(1,2, figsize=(12,5))
    #ax = axes[0]
    #h1 = ax.hist(model_mean, bins=xbins, color=(1.0,0.,0.,0.4), label='MOD', log=True)
    #ax.set_xlabel('wind [m/s]')
    #ax.set_title('MODEL (red) and OBS (grey)')
    #h2 = ax.hist(obs_mean, bins=xbins, color=(0.1,0.1,0.1,0.4), label='OBS', log=True)
    ##ax.legend([h1, h2], ['MOD', 'OBS'])
    #ax = axes[1]
    #ax.hist((model_mean - obs_mean), bins=diffbins, log=True)
    #ax.set_xlabel('wind difference [m/s]')
    #ax.set_title('MODEL - OBS')
    #plt.show()
    #quit()
    #####################



    ##################### SPAGHETTI PLOT
    #fig, axes = plt.subplots(1,2, figsize=(18,5))
    #for i in range(30):
    #    ax = axes[1]
    #    line1, = ax.plot(dates, model_mean[:,i], color='red', label='model')
    #    line2, = ax.plot(dates, -obs_mean[:,i], color='black', label='obs')
    #    if i == 0:
    #        ax.legend([line1, line2], ['model', 'obs'])
    #    ax = axes[0]
    #    line1, = ax.plot(dates, model_mean[:,i] - obs_mean[:,i], color='black', label='model error')
    #    if i == 0:
    #        ax.legend([line1], ['model error'])

    #model_mean_med = np.nanpercentile(model_mean, q=90, axis=1)
    #obs_mean_med = np.nanpercentile(obs_mean, q=90, axis=1)
    #diff_mean_med = np.nanmedian(model_mean - obs_mean, axis=1)

    #ax = axes[1]
    #ax.plot(dates, model_mean_med, color='orange', lineWidth=3)
    #ax.plot(dates, -obs_mean_med, color='orange', lineWidth=3)
    #ax.axhline(y=0)
    #ax.set_xlabel('date')
    #ax.set_ylabel('wind [m/s]')
    #ax.set_ylim((-30,30))
    #ax = axes[0]
    #ax.plot(dates, -diff_mean_med, color='orange', lineWidth=3)
    #ax.axhline(y=0)
    #ax.set_xlabel('date')
    #ax.set_ylabel('wind error [m/s]')
    #ax.set_ylim((-20,20))
    #plt.show()
    #####################

    ##################### ERROR SCATTER
    #err_mean = model_mean - obs_mean
    #ax = plt.gca()
    #ax.scatter(obs_mean, err_mean)
    #ax.axhline(y=0, color='k')
    #plt.show()
    #####################

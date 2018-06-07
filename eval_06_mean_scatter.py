import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from sklearn.linear_model import LinearRegression
from functions import calc_model_fields, calc_scores
import globals as G
from filter import EntryFilter


############ USER INPUT #############
# obs case name (name of obs pkl file in data folder)
obs_case_name = 'burglind'
#obs_case_name = 'foehn_apr18'
# model case name (name of folder with model data in 'mod_path'
model_case_name = 'burglind_ref'
#model_case_name = 'foehn_apr18_ref'
# obs model combination case name
obs_model_case_name = 'OBS_'+obs_case_name+'_MODEL_'+model_case_name
# do not plot (0) show plot (1) save plot (2)
i_plot = 2
# model fields to calculate and plot
i_model_fields = [G.MEAN_WIND]
i_scores = [G.SCORE_ME]
# path of input obs_model pickle file
data_pickle_path = '../data/'+obs_model_case_name+'.pkl'
plot_base_dir = '../plots/'
plot_case_dir = plot_base_dir + obs_model_case_name + '/'
#####################################

# create directories
if i_plot > 1 and not os.path.exists(plot_case_dir):
    os.mkdir(plot_case_dir)

# load data
data = pickle.load( open(data_pickle_path, 'rb') )

# calculate gusts
data = calc_model_fields(data, i_model_fields)
data = calc_scores(data, i_scores)

station_names = np.asarray(data[G.STAT_NAMES])

nstat = len(station_names)
nts = len(data[G.OBS][G.DTS])

## Aggregate over stations
obs_wind = np.zeros((nts,nstat))
mod_err_dict = {}
for si,stat in enumerate(station_names):
    obs_wind[:,si] = data[G.OBS][G.STAT][stat][G.PAR]['FF_10M'].values

for field_name in i_model_fields:
    mod_err = np.zeros((nts,nstat))
    for si,stat in enumerate(station_names):
        mod_err[:,si] = data[G.MODEL][G.STAT][stat][G.SCORE][field_name][G.SCORE_ME]
    mod_err_dict[field_name] = mod_err



stat = 'SAE'
si = np.argwhere(station_names == stat).squeeze()

si = np.arange(0,len(station_names))
title = 'all 512 stations'
#print(si)

###################### PLOT model error vs obs gust
if i_plot > 0:
    # regression
    reg = LinearRegression(fit_intercept=True)

    # plot preparation
    #fig,axes = plt.subplots(2,3,figsize=(14,8))
    fig = plt.figure(figsize=(14,8))
    nrow = 1
    ncol = 1
    ymax = -np.Inf
    ymin = np.Inf

    # loop over axes and mean wind
    axes = []
    for mi,field_name in enumerate(i_model_fields):
        print(field_name)
        ax = fig.add_subplot(nrow, ncol, mi+1)
        axes.append(ax)

        # prepare feature matrix
        X = obs_wind[:,si].flatten().reshape(-1,1)
        y = mod_err_dict[field_name][:,si].flatten()

        # delete NAN
        size_before = y.shape[0]
        mask = np.isnan(X[:,0])
        X = X[~mask,:]
        y = y[~mask]
        size_after = y.shape[0]
        print('\t deleted ' + str(np.round(1 - size_after/size_before,3)) + ' % of entries due to NaN.')

        # determine max/min y
        ymax = max(np.max(y),ymax)
        ymin = min(np.min(y),ymin)

        # fit regression and draw line
        reg.fit(X,y)
        line = reg.predict(X)

        # plotting
        ax.scatter(obs_wind[:,si], mod_err_dict[field_name][:,si], color='black', marker=".")
        ax.plot(X, line, color='red')
        ax.set_xlabel('Observed mean wind (OBS) [m/s]')
        if mi == 0:
            ax.set_ylabel('Model absolute error in mean wind (MOD-OBS) [m/s]')
        ax.axhline(0, color='k', linewidth=0.8)
        ax.set_title(field_name)
    # set axes limits in each ax
    for ax in axes:
        ax.set_ylim((ymin,ymax))

    # finish plot
    plt.suptitle(title)
    plt.subplots_adjust(left=0.05,bottom=0.08,right=0.95,top=0.9,wspace=0.2,hspace=0.3)

    if i_plot == 1:
        plt.show()
    elif i_plot > 1:
        plot_name = plot_case_dir + 'scatter_mean_wind.png'
        plt.savefig(plot_name)
        plt.close('all')



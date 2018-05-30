import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
#from sklearn import metrics
from sklearn.linear_model import LinearRegression
from functions import calc_gusts, calc_scores
import globals as G
from filter import EntryFilter


############ USER INPUT #############
# obs case name (name of obs pkl file in data folder)
obs_case_name = 'burglind'
obs_case_name = 'foehn_apr18'
# model case name (name of folder with model data in 'mod_path'
model_case_name = 'burglind_ref'
model_case_name = 'foehn_apr18_ref'
# obs model combination case name
obs_model_case_name = 'OBS_'+obs_case_name+'_MODEL_'+model_case_name
# do not plot (0) show plot (1) save plot (2)
i_plot = 1
# gust methods to calculate and plot
i_gust_fields = [G.GUST_MIX_COEF_LINEAR,
                G.GUST_MIX_COEF_NONLIN,
                G.GUST_BRASSEUR_ESTIM,
                G.GUST_BRASSEUR_UPBOU,
                G.GUST_BRASSEUR_LOBOU]
i_scores = [G.SCORE_ME]
# path of input obs_model pickle file
data_pickle_path = '../data/'+obs_model_case_name+'.pkl'
plot_base_dir = '../plots/'
plot_case_dir = plot_base_dir + obs_model_case_name + '/'
min_gust = 10
#####################################

EF = EntryFilter()

# create directories
if i_plot > 1 and not os.path.exists(plot_case_dir):
    os.mkdir(plot_case_dir)

# load data
data = pickle.load( open(data_pickle_path, 'rb') )

# calculate gusts
data = calc_gusts(data, i_gust_fields)
# filter according to min gust strength
data = EF.filter_according_obs_gust(data, min_gust)
#data = remove_obs_nan_in_hourly_fields(data, 'VMAX_10M1')
data = calc_scores(data, i_scores)

station_names = np.asarray(data[G.STAT_NAMES])

nstat = len(station_names)
nts = len(data[G.OBS][G.DTS])

## Aggregate over stations
obs_gust = np.zeros((nts,nstat))
mod_err_dict = {}
for si,stat in enumerate(station_names):
    obs_gust[:,si] = data[G.OBS][G.STAT][stat][G.PAR]['VMAX_10M1'].values

for method in i_gust_fields:
    mod_err = np.zeros((nts,nstat))
    for si,stat in enumerate(station_names):
        mod_err[:,si] = data[G.MODEL][G.STAT][stat][G.SCORE][method][G.SCORE_ME]
    mod_err_dict[method] = mod_err



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
    nrow = 2
    ncol = 3
    ymax = -np.Inf
    ymin = np.Inf

    # loop over axes and gust calc method
    axes = []
    for mi,method in enumerate(i_gust_fields):
        print(method)
        #ax = axes[mi]
        ax = fig.add_subplot(nrow, ncol, mi+1)
        axes.append(ax)

        # prepare feature matrix
        X = obs_gust[:,si].flatten().reshape(-1,1)
        y = mod_err_dict[method][:,si].flatten()

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
        ax.scatter(obs_gust[:,si], mod_err_dict[method][:,si], color='black', marker=".")
        ax.plot(X, line, color='red')
        ax.set_xlabel('Observed gust (OBS) [m/s]')
        if mi == 0:
            ax.set_ylabel('Model absolute error (MOD-OBS) [m/s]')
        ax.axhline(0, color='k', linewidth=0.8)
        ax.set_title(method)
    # set axes limits in each ax
    for ax in axes:
        ax.set_ylim((ymin,ymax))

    # finish plot
    plt.suptitle(title)
    plt.subplots_adjust(left=0.05,bottom=0.08,right=0.95,top=0.9,wspace=0.2,hspace=0.3)

    if i_plot == 1:
        plt.show()
    elif i_plot > 1:
        plot_name = plot_case_dir + 'scatter.png'
        plt.savefig(plot_name)
        plt.close('all')


quit()
####################### CALCULATE ERROR MEASURES
## CONTINUOUS ERROR MEASURES
#error_measures = {}
#error_measures['mean_abs_err'] = np.full((nstat,nmethods), np.nan)
#error_measures['r2'] = np.full((nstat,nmethods), np.nan)
#error_measures['explained_var'] = np.full((nstat,nmethods), np.nan)
##sis = np.arange(0,600)
#for si in range(0,nstat):
#    y_obs = obs_gust[:,si]
#
#    # find nans
#    mask = np.isnan(y_obs)
#    y_obs = y_obs[~mask]
#
#    for i,mi in enumerate(i_methods):
#        y_mod = mod_gust[:,si,mi-1].flatten()
#        y_mod = y_mod[~mask]
#        error_measures['mean_abs_err'][si,i] = metrics.mean_absolute_error(y_obs, y_mod)
#        error_measures['r2'][si,i] = metrics.r2_score(y_obs, y_mod)
#        error_measures['explained_var'][si,i] = metrics.explained_variance_score(y_obs, y_mod)
#
#
## CATEGORICAL ERROR MEASURES
#
#
## print measures to file
#file_name = plot_case_dir + 'scores.txt'
#with open(file_name, 'w') as f:
#    f.write('i_methods ' + str(i_methods) + '\n')
#    for key,meas in error_measures.items():
#        mean_meas = np.mean(meas, axis=0)
#        text = 'station mean ' + key + ' ' + str(mean_meas)
#        print(text)
#        f.write(text + '\n')
#
#
####################### PLOT CONTINUOUS ERROR MEASURES
#if i_plot > 0:
#    fig,axes = plt.subplots(1,3,figsize=(14,4))
#    i = 0
#    for key,meas in error_measures.items():
#        ax = axes[i]
#        if key == 'mean_abs_err':
#            bins = np.arange(0,20.1,1)
#        else:
#            bins = np.arange(-1,1.1,0.1)
#        ax.hist(meas,bins=bins, histtype='step',
#                color=['red','orange','black'], label=['Method 1', 'Method 2', 'Method 4'])
#        ax.axvline(0,color='k')
#        if key == 'mean_abs_err':
#            ax.legend(loc='upper right')
#            ax.set_xlabel('Absolute error [m/s]')
#        else:
#            ax.legend(loc='upper left')
#            ax.set_xlabel(']-inf,1]')
#        ax.set_title(key)
#        i += 1
#
#    if i_plot == 1:
#        plt.show()
#    elif i_plot > 1:
#        plot_name = plot_case_dir + 'err_continuous.png'
#        plt.savefig(plot_name)
#        plt.close('all')


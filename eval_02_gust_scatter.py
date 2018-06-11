import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from sklearn.linear_model import LinearRegression
from functions import calc_model_fields, join_model_and_obs, calc_scores
import globals as G
from filter import EntryFilter
from namelist_cases import Case_Namelist

############ USER INPUT #############
case_index = 0
CN = Case_Namelist(case_index)
i_plot = 1
# gust methods to calculate and plot
i_model_fields = [G.GUST_MIX_COEF_LINEAR,
                G.GUST_MIX_COEF_NONLIN,
                G.GUST_BRASSEUR_ESTIM,
                G.GUST_BRASSEUR_LOBOU,
                G.GUST_BRASSEUR_UPBOU]
i_scores = [G.SCORE_ME]
brasseur_gusts = [G.GUST_BRASSEUR_ESTIM,
                G.GUST_BRASSEUR_LOBOU,
                G.GUST_BRASSEUR_UPBOU]
# path of input obs_model pickle file
min_gust_levels = [0,10,20]
min_gust_levels = [20]
#####################################

# create directories
if i_plot > 1 and not os.path.exists(CN.plot_path):
    os.mkdir(CN.plot_path)

EF = EntryFilter()

for min_gust in min_gust_levels:

    # load data
    data = pickle.load( open(CN.mod_path, 'rb') )

    # calculate gusts
    data = calc_model_fields(data, i_model_fields)
    # join model and obs
    data = join_model_and_obs(data)

    # filter according to min gust strength
    data = EF.filter_according_obs_gust(data, min_gust)
    data = calc_scores(data, i_scores)

    station_names = np.asarray(data[G.STAT_NAMES])

    nstat = len(station_names)
    nts = len(data[G.OBS][G.DTS])

    ## Aggregate over stations
    obs_gust = np.zeros((nts,nstat))
    mod_err_dict = {}
    for si,stat in enumerate(station_names):
        obs_gust[:,si] = data[G.OBS][G.STAT][stat][G.PAR]['VMAX_10M1'].values

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
        nrow = 2
        ncol = 3
        ymax = -np.Inf
        ymin = np.Inf

        # loop over axes and gust calc method
        axes = []
        for mi,field_name in enumerate(i_model_fields):
            print(field_name)
            #ax = axes[mi]
            if field_name in brasseur_gusts:
                ax = fig.add_subplot(nrow, ncol, mi+2)
            else:
                ax = fig.add_subplot(nrow, ncol, mi+1)
            axes.append(ax)

            # prepare feature matrix
            X = obs_gust[:,si].flatten().reshape(-1,1)
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
            ax.scatter(obs_gust[:,si], mod_err_dict[field_name][:,si], color='black', marker=".")
            ax.plot(X, line, color='red')
            ax.set_xlabel('Observed gust (OBS) [m/s]')
            if mi == 0:
                ax.set_ylabel('Model absolute error (MOD-OBS) [m/s]')
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
            plot_name = plot_case_dir + 'scatter_minGust_'+str(min_gust).zfill(2)+'.png'
            plt.savefig(plot_name)
            plt.close('all')



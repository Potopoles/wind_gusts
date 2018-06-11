import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from sklearn.linear_model import LinearRegression
from functions import calc_model_fields, calc_scores
import globals as G
from filter import StationFilter, EntryFilter


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
# gust methods to calculate and plot
i_gust_fields = [G.GUST_MIX_COEF_LINEAR,
                G.GUST_MIX_COEF_NONLIN,
                G.GUST_BRASSEUR_ESTIM]
i_scores = [G.SCORE_ME]
# path of input obs_model pickle file
data_pickle_path = '../data/'+obs_model_case_name+'.pkl'
plot_base_dir = '../plots/'
plot_case_dir = plot_base_dir + obs_model_case_name + '/'
min_gust_levels = [0,10,20]
#min_gust_levels = [10]
#altitudes = [[0,1000],
#            [1001,2000],
#            [2001,4000]]
altitudes = [[0,800],
            [801,2000],
            [2001,4000]]
i_plot_alt_hist = 1
#####################################


# altitude histogram
if i_plot_alt_hist:
    data = pickle.load( open(data_pickle_path, 'rb') )
    station_names = data[G.STAT_NAMES]
    alts = np.zeros(len(station_names))
    for i,stat in enumerate(station_names):
        alts[i] = data[G.OBS][G.STAT][stat][G.STAT_META]['Height'].values
    plt.hist(alts, bins=50)
    plot_name = plot_case_dir + 'alt_stations.png'
    plt.savefig(plot_name)
    plt.close('all')

# create directories
if i_plot > 1 and not os.path.exists(plot_case_dir):
    os.mkdir(plot_case_dir)

EF = EntryFilter()

for min_gust in min_gust_levels:

    # load data
    data = pickle.load( open(data_pickle_path, 'rb') )

    # filter stations according to tags
    SF = StationFilter()
    filtered = SF.filter_according_altitude(data, altitudes)
    tags = list(filtered.keys())
    #for stat in filtered[tags[1]][G.STAT_NAMES]:
    #    print(filtered[tags[1]][G.OBS][G.STAT][stat][G.STAT_META]['Height'].values)

    obs_gust_tag = {}
    mod_err_dict_tag = {}
    stations_filtered_tag = {}
    for tag in tags:
        print(tag)
        ## calculate gusts and scores
        filtered[tag] = calc_model_fields(filtered[tag], i_gust_fields)
        filtered[tag] = EF.filter_according_obs_gust(filtered[tag], min_gust)
        filtered[tag] = calc_scores(filtered[tag], i_scores)

        stations_filtered = filtered[tag][G.STAT_NAMES]
        nstat = len(stations_filtered)
        nts = len(filtered[tag][G.OBS][G.DTS])

        ## Aggregate over stations
        obs_gust = np.zeros((nts,nstat))
        mod_err_dict = {}
        for si,stat in enumerate(stations_filtered):
            obs_gust[:,si] = filtered[tag][G.OBS][G.STAT][stat][G.PAR]['VMAX_10M1'].values

        for method in i_gust_fields:
            mod_err = np.zeros((nts,nstat))
            for si,stat in enumerate(stations_filtered):
                mod_err[:,si] = filtered[tag][G.MODEL][G.STAT][stat][G.SCORE][method][G.SCORE_ME]
            mod_err_dict[method] = mod_err

        obs_gust_tag[tag] = obs_gust
        mod_err_dict_tag[tag] = mod_err_dict
        stations_filtered_tag[tag] = stations_filtered


    ###################### PLOT model error vs obs gust
    if i_plot > 0:
        # regression
        reg = LinearRegression(fit_intercept=True)

        # plot preparation
        ncol = len(tags)
        nrow = len(i_gust_fields)
        fig,axes = plt.subplots(nrow,ncol,figsize=(ncol*3.8,nrow*3.3))
        #fig = plt.figure(figsize=(14,8))
        #nrow = len(tags)
        #ncol = len(i_gust_fields)
        ymax = -np.Inf
        ymin = np.Inf

        # loop over axes and gust calc method
        for ti,tag in enumerate(tags):

            obs_gust = obs_gust_tag[tag]
            mod_err_dict = mod_err_dict_tag[tag]
            si = np.arange(0,len(stations_filtered_tag[tag]))

            for mi,method in enumerate(i_gust_fields):
                ax = axes[mi][ti]

                # prepare feature matrix
                X = obs_gust[:,si].flatten().reshape(-1,1)
                y = mod_err_dict[method][:,si].flatten()

                # delete NAN
                size_before = y.shape[0]
                mask = np.isnan(X[:,0])
                X = X[~mask,:]
                y = y[~mask]
                size_after = y.shape[0]

                # determine max/min y
                if len(y) > 0:
                    ymax = max(np.max(y),ymax)
                    ymin = min(np.min(y),ymin)

                    # fit regression and draw line
                    reg.fit(X,y)
                    line = reg.predict(X)

                    # plotting
                    ax.scatter(obs_gust[:,si], mod_err_dict[method][:,si], color='black', marker=".")
                    ax.plot(X, line, color='red')
                    if mi == len(i_gust_fields)-1:
                        ax.set_xlabel('Observed gust (OBS) [m/s]')
                    if ti == 0:
                        ax.set_ylabel('Model absolute error (MOD-OBS) [m/s]')
                    ax.axhline(0, color='k', linewidth=0.8)
                    ax.set_title(method + '  ' + tag)

        # set axes limits in each ax
        for axs in axes:
            for ax in axs:
                ax.set_ylim((ymin,ymax))

        # finish plot
        plt.suptitle(obs_model_case_name)
        plt.subplots_adjust(left=0.10,bottom=0.08,right=0.98,top=0.92,wspace=0.15,hspace=0.25)

        if i_plot == 1:
            plt.show()
        elif i_plot > 1:
            plot_name = plot_case_dir + 'altitude_tags_minGust_'+str(min_gust).zfill(2)+'.png'
            print(plot_name)
            plt.savefig(plot_name)
            plt.close('all')


import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from sklearn.linear_model import LinearRegression
from functions import calc_model_fields, join_model_and_obs, \
                        join_model_runs, join_all_stations
import globals as G
from filter import StationFilter, EntryFilter
from namelist_cases import Case_Namelist


############ USER INPUT #############
case_index = 3
CN = Case_Namelist(case_index)
# do not plot (0) show plot (1) save plot (2)
i_plot = 2
# model fields to calculate 
i_model_fields = [G.GUST_MIX_COEF_LINEAR,
                G.GUST_MIX_COEF_NONLIN,
                G.GUST_BRASSEUR_ESTIM]
min_gust_levels = [0,5,10,20]
#min_gust_levels = [18]
altitudes = [[0,1000],
            [1001,2000],
            [2001,4000]]
#altitudes = [[0,800],
#            [801,2000],
#            [2001,4000]]
i_plot_alt_hist = 2
#####################################

# create directories
if ((i_plot > 1) or (i_plot_alt_hist > 1))  and not os.path.exists(CN.plot_path):
    os.mkdir(CN.plot_path)

# string for the current altitudes setting
alt_str = ""
for alt in altitudes[:-1]:
    alt_str = alt_str + str(alt[1]) + '_'
alt_str = alt_str[:-1]

# altitude histogram
if i_plot_alt_hist:
    data = pickle.load( open(CN.mod_path, 'rb') )
    station_names = data[G.STAT_NAMES]
    alts = np.zeros(len(station_names))
    for i,stat in enumerate(station_names):
        alts[i] = data[G.STAT_META][stat]['Height'].values
    fig,axes = plt.subplots(figsize=(8,4))
    plt.hist(alts, bins=50)
    plt.grid()
    plt.xlabel('Altitude [m]')
    plt.ylabel('Number of Stations')
    plt.title('Station Altitude Histogram')
    for alt in altitudes:
        plt.axvline(x=alt[0], color='k')
        plt.axvline(x=alt[1], color='k')
    if i_plot_alt_hist == 1:
        plt.show()
    elif i_plot_alt_hist > 1:
        plot_name = CN.plot_path + 'altitude_stations_('+alt_str+').png'
        plt.savefig(plot_name)
        plt.close('all')


EF = EntryFilter()

for min_gust in min_gust_levels:
    print('########## ' + str(min_gust) + ' ##########')

    # load data
    data = pickle.load( open(CN.mod_path, 'rb') )

    # filter stations according to altitudes
    SF = StationFilter()
    filtered = SF.filter_according_altitude(data, altitudes)
    tags = list(filtered.keys())

    for tag in tags:
        print('##### ' + tag + ' #####')
        tag_data = filtered[tag]
        # calculate gusts
        tag_data = calc_model_fields(tag_data, i_model_fields)
        # join model and obs
        tag_data = join_model_and_obs(tag_data)
        # filter according to min gust strength
        tag_data = EF.filter_according_obs_gust(tag_data, min_gust)
        # join all model runs
        tag_data = join_model_runs(tag_data)
        # join stations
        tag_data = join_all_stations(tag_data)
        
        filtered[tag] = tag_data

    ###################### PLOT
    if i_plot > 0:
        # regression
        reg = LinearRegression(fit_intercept=True)

        # plot preparation
        ncol = len(tags)
        nrow = len(i_model_fields)
        fig,axes = plt.subplots(nrow,ncol,figsize=(ncol*3.8,nrow*3.3))

        ymax = -np.Inf
        ymin = np.Inf

        # loop over axes and gust calc field_name
        for ti,tag in enumerate(tags):

            df = filtered[tag][G.BOTH][G.ALL_STAT]

            for mi,field_name in enumerate(i_model_fields):
                ax = axes[mi][ti]

                # prepare feature matrix
                X = df[G.OBS_GUST_SPEED].values.reshape(-1,1)
                y = (df[field_name] - df[G.OBS_GUST_SPEED]).values

                # delete NAN
                mask = np.isnan(X[:,0])
                X = X[~mask,:]
                y = y[~mask]

                # determine max/min y
                if len(y) > 0:
                    ymax = max(np.max(y),ymax)
                    ymin = min(np.min(y),ymin)

                    # fit regression and draw line
                    reg.fit(X,y)
                    line = reg.predict(X)

                    # plotting
                    ax.scatter(X[:,0], y, color='black', marker=".")
                    ax.plot(X, line, color='red')
                    if mi == len(i_model_fields)-1:
                        ax.set_xlabel('Observed gust (OBS) [m/s]')
                    if ti == 0:
                        ax.set_ylabel('Model absolute error (MOD-OBS) [m/s]')
                    ax.axhline(0, color='k', linewidth=0.8)
                    ax.set_title(field_name + '  ' + tag)

        # set axes limits in each ax
        for axs in axes:
            for ax in axs:
                ax.set_ylim((ymin,ymax))

        # finish plot
        title = 'altitude sep ' + alt_str + ' ' + CN.case_name + ' minGust ' + str(min_gust) +' m/s'
        plt.suptitle(title)
        plt.subplots_adjust(left=0.10,bottom=0.08,right=0.98,top=0.92,wspace=0.15,hspace=0.25)

        if i_plot == 1:
            plt.show()
        elif i_plot > 1:
            plot_name = CN.plot_path + 'altitude_tags_'+alt_str+'_minGust_'+str(min_gust).zfill(2)+'.png'
            print(plot_name)
            plt.savefig(plot_name)
            plt.close('all')



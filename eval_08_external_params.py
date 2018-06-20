import os
import numpy as np
import pandas as pd
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
case_index = 0
CN = Case_Namelist(case_index)
# do not plot (0) show plot (1) save plot (2)
i_plot = 2
# model fields to calculate 
i_model_fields = [G.GUST_MIX_COEF_LINEAR,
                G.GUST_MIX_COEF_NONLIN,
                G.MODEL_MEAN_WIND,
                G.GUST_BRASSEUR_ESTIM,
                G.GUST_BRASSEUR_LOBOU,
                G.GUST_ICON]
min_gust_levels = [0,5,10,20]
min_gust_levels = [5]
varnames = ['dz','Height','sso_stdh','slo_ang',
            'skyview','slo_asp','z0']
#varname = 'dz'
#varname = 'Height'
#varname = 'sso_stdh'
#varname = 'slo_ang'
#varname = 'skyview'
#varname = 'slo_asp'
#varname = 'z0'
#####################################


# create directories
if i_plot > 1 and not os.path.exists(CN.plot_path):
    os.mkdir(CN.plot_path)

EF = EntryFilter()

for min_gust in min_gust_levels:
    print('########## ' + str(min_gust) + ' ##########')

    for varname in varnames:
        print('###### ' + str(varname) + ' ######')

        # load data
        data = pickle.load( open(CN.mod_path, 'rb') )


        # calculate gusts
        data = calc_model_fields(data, i_model_fields)

        # join model and obs
        data = join_model_and_obs(data)
        # filter according to min gust strength
        data = EF.filter_according_obs_gust(data, min_gust)
        # join all model runs
        data = join_model_runs(data)

        # Add station variable to combined time series
        for stat_key in data[G.STAT_NAMES]:
            both = data[G.BOTH][G.STAT][stat_key]
            series = pd.Series(data[G.STAT_META][stat_key][varname].values[0], index=both.index)
            both[varname] = series
            data[G.BOTH][G.STAT][stat_key] = both

        # Take average over stations
        for stat_key in data[G.STAT_NAMES]:
            both = data[G.BOTH][G.STAT][stat_key]
            both = both.mean(0)
            data[G.BOTH][G.STAT][stat_key] = both

        # join stations
        data = join_all_stations(data)


        ###################### PLOT
        if i_plot > 0:
            df = data[G.BOTH][G.ALL_STAT]

            # regression
            reg = LinearRegression(fit_intercept=True)

            # plot preparation
            fig = plt.figure(figsize=(14,9))
            nrow = 2
            ncol = 3
            ymax = -np.Inf
            ymin = np.Inf

            # loop over axes and gust calc method
            axes = []
            for mi,field_name in enumerate(i_model_fields):
                print(field_name)

                ax = fig.add_subplot(nrow, ncol, mi+1)
                axes.append(ax)

                # prepare feature matrix
                if field_name == G.MODEL_MEAN_WIND:
                    X = df[G.OBS_MEAN_WIND].values.reshape(-1,1)
                    y = df[field_name].values - df[G.OBS_MEAN_WIND].values
                    col = df[varname].values
                    xlab = 'Observed mean wind (OBS) [m/s]'
                    ylab = 'Model mean wind error (MOD-OBS) [m/s]'
                else:
                    X = df[G.OBS_GUST_SPEED].values.reshape(-1,1)
                    y = df[field_name].values - df[G.OBS_GUST_SPEED].values
                    col = df[varname].values
                    xlab = 'Observed gust (OBS) [m/s]'
                    ylab = 'Model gust error (MOD-OBS) [m/s]'


                # delete NAN
                mask = np.isnan(X[:,0])
                mask[np.isnan(y)] = True
                X = X[~mask,:]
                y = y[~mask]
                col = col[~mask]


                # determine max/min y
                ymax = max(np.max(y),ymax)
                ymin = min(np.min(y),ymin)

                # calculate median
                dmp = 1
                mp_borders = np.arange(np.floor(ymin),np.ceil(ymax),dmp)
                mp_x = mp_borders[:-1] + np.diff(mp_borders)/2
                mp_y = np.full(len(mp_x),np.nan)
                for i in range(0,len(mp_x)):
                    #inds = np.squeeze(np.argwhere((X[:,0] > mp_borders[i]) & (X[:,0] <= mp_borders[i+1])))
                    inds = (X[:,0] > mp_borders[i]) & (X[:,0] <= mp_borders[i+1])
                    if np.sum(inds) > 20:
                        mp_y[i] = np.median(y[inds])
                    

                # fit regression and draw line
                reg.fit(X,y)
                line = reg.predict(X)

                # plotting
                ax.scatter(X[:,0], y, c=col, marker=".")
                ax.plot(X, line, color='red')
                ax.plot(mp_x, mp_y, color='orange')
                ax.set_xlabel(xlab)
                if mi % ncol == 0 or field_name == G.MODEL_MEAN_WIND:
                    ax.set_ylabel(ylab)
                ax.axhline(0, color='k', linewidth=0.8)
                ax.set_title(field_name)
                if mi == 0:
                    ax.text(np.max(X)-0.13*(np.max(X)-np.min(X)), ymax-0.10*(ymax-ymin), 'n '+str(len(y)))
                # correlation
                corr = 'cor '+str(np.round(np.corrcoef(y,col)[1][0],2))
                ax.text(np.max(X)-0.33*(np.max(X)-np.min(X)), ymax-0.10*(ymax-ymin), corr)

            # set axes limits in each ax
            for i,ax in enumerate(axes):
                ax.set_ylim((ymin,ymax))
                ax.set_xlim(left=min_gust)

            # finish plot
            title = CN.case_name + ' minGust ' + str(min_gust) +' m/s'
            plt.suptitle(title)
            plt.subplots_adjust(left=0.05,bottom=0.08,right=0.95,top=0.9,wspace=0.2,hspace=0.3)

            if i_plot == 1:
                plt.show()
            elif i_plot > 1:
                plot_name = CN.plot_path + 'extPar_minGust_'+str(min_gust).zfill(2)+'_'+varname+'.png'
                print(plot_name)
                plt.savefig(plot_name)
                plt.close('all')



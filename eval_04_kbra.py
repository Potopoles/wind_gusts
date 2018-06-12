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
import copy
from namelist_cases import Case_Namelist


############ USER INPUT #############
case_index = 1
CN = Case_Namelist(case_index)
# do not plot (0) show plot (1) save plot (2)
i_plot = 2
# model fields to calculate 
i_gust_fields = [G.GUST_BRASSEUR_ESTIM,
                G.GUST_BRASSEUR_UPBOU,
                G.GUST_BRASSEUR_LOBOU]
i_k_fields =   [G.KVAL_BRASSEUR_ESTIM,
                G.KVAL_BRASSEUR_LOBOU,
                G.KVAL_BRASSEUR_UPBOU]
min_gust_levels = [0,10,20]
#min_gust_levels = [18]
#####################################

# create directories
if i_plot > 1 and not os.path.exists(CN.plot_path):
    os.mkdir(CN.plot_path)

EF = EntryFilter()

for min_gust in min_gust_levels:
    print('########## ' + str(min_gust) + ' ##########')

    # load data
    data = pickle.load( open(CN.mod_path, 'rb') )

    # calculate gusts
    i_model_fields = copy.copy(i_gust_fields)
    i_model_fields.extend(i_k_fields)

    # calculate gusts
    data = calc_model_fields(data, i_model_fields)
    # join model and obs
    data = join_model_and_obs(data)
    # filter according to min gust strength
    data = EF.filter_according_obs_gust(data, min_gust)
    # join all model runs
    data = join_model_runs(data)
    # join stations
    data = join_all_stations(data)

    ###################### PLOT model error vs obs gust
    if i_plot > 0:
        df = data[G.BOTH][G.ALL_STAT]

        # regression
        reg = LinearRegression(fit_intercept=True)

        # plot preparation
        fig,axes = plt.subplots(1,3,figsize=(14,4))

        ymax = -np.Inf
        ymin = np.Inf

        # loop over axes and gust calc method
        for i in range(0,len(i_gust_fields)):
            ax = axes[i]

            # prepare feature matrix
            X = (df[i_gust_fields[i]] - df[G.OBS_GUST_SPEED]).values.reshape(-1,1)
            y = df[i_k_fields[i]].values

            # delete NAN
            mask = np.isnan(X[:,0])
            X = X[~mask,:]
            y = y[~mask]

            # determine max/min y
            ymax = max(np.max(y),ymax)
            ymin = min(np.min(y),ymin)

            # fit regression and draw line
            reg.fit(X,y)
            line = reg.predict(X)

            ax.scatter(X[:,0], y, color='black', marker=".")
            ax.plot(X, line, color='red')
            ax.set_xlabel('Model absolute error (MOD-OBS) [m/s]')
            if i == 0:
                ax.set_ylabel('model level of Brasseur wind')
            ax.axhline(0, color='k', linewidth=0.8)
            ax.set_title(i_gust_fields[i])
        # set axes limits in each ax
        for ax in axes:
            ax.set_ylim((ymin,ymax))

        # finish plot
        title = CN.case_name + 'brasseur model level minGust ' + str(min_gust) +' m/s'
        plt.suptitle(title)
        plt.subplots_adjust(left=0.05,bottom=0.12,right=0.95,top=0.85,wspace=0.2,hspace=0.3)

        if i_plot == 1:
            plt.show()
        elif i_plot > 1:
            plot_name = CN.plot_path + 'kbra_minGust_'+str(min_gust).zfill(2)+'.png'
            print(plot_name)
            plt.savefig(plot_name)
            plt.close('all')



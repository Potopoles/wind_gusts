import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from sklearn.linear_model import LinearRegression
from functions import calc_gusts, calc_scores
import globals as G
from filter import EntryFilter
import copy


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
i_plot = 2
# gust methods to calculate and plot
i_gust_fields = [G.GUST_BRASSEUR_ESTIM,
                G.GUST_BRASSEUR_UPBOU,
                G.GUST_BRASSEUR_LOBOU]
i_k_fields =   [G.KVAL_BRASSEUR_ESTIM,
                G.KVAL_BRASSEUR_LOBOU,
                G.KVAL_BRASSEUR_UPBOU]
i_scores = [G.SCORE_ME]
# path of input obs_model pickle file
data_pickle_path = '../data/'+obs_model_case_name+'.pkl'
plot_base_dir = '../plots/'
plot_case_dir = plot_base_dir + obs_model_case_name + '/'
min_gust_levels = [0,10,20]
#####################################


# create directories
if i_plot > 1 and not os.path.exists(plot_case_dir):
    os.mkdir(plot_case_dir)

EF = EntryFilter()

for min_gust in min_gust_levels:

    # load data
    data = pickle.load( open(data_pickle_path, 'rb') )

    # calculate gusts
    all_fields = copy.copy(i_gust_fields)
    all_fields.extend(i_k_fields)
    data = calc_gusts(data, all_fields)
    # filter according to min gust strength
    data = EF.filter_according_obs_gust(data, min_gust)
    data = calc_scores(data, i_scores)

    station_names = np.asarray(data[G.STAT_NAMES])
    nstat = len(station_names)
    nts = len(data[G.OBS][G.DTS])

    ## Aggregate over stations
    mod_err_dict = {}
    k_val_dict = {}

    for i in range(0,len(i_gust_fields)):
        mod_err = np.zeros((nts,nstat))
        k_val = np.zeros((nts,nstat))
        for si,stat in enumerate(station_names):
            mod_err[:,si] = data[G.MODEL][G.STAT][stat][G.SCORE][i_gust_fields[i]][G.SCORE_ME]
            k_val[:,si] = data[G.MODEL][G.STAT][stat][G.GUST][i_k_fields[i]]
        mod_err_dict[i_gust_fields[i]] = mod_err
        k_val_dict[i_k_fields[i]] = k_val



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
        fig,axes = plt.subplots(1,3,figsize=(14,4))

        ymax = -np.Inf
        ymin = np.Inf

        # loop over axes and gust calc method
        for i in range(0,len(i_gust_fields)):
            ax = axes[i]

            # prepare feature matrix
            X = mod_err_dict[i_gust_fields[i]][:,si].flatten().reshape(-1,1)
            y = k_val_dict[i_k_fields[i]][:,si].flatten()

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
            ax.scatter(mod_err_dict[i_gust_fields[i]][:,si],
                       k_val_dict[i_k_fields[i]][:,si],
                        color='black', marker=".")
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
        plt.suptitle(title)
        plt.subplots_adjust(left=0.05,bottom=0.12,right=0.95,top=0.9,wspace=0.2,hspace=0.3)

        if i_plot == 1:
            plt.show()
        elif i_plot > 1:
            plot_name = plot_case_dir + 'kbra_minGust_'+str(min_gust).zfill(2)+'.png'
            plt.savefig(plot_name)
            plt.close('all')



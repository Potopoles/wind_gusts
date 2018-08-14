import os
import numpy as np
import matplotlib as mpl
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
                G.GUST_ICON,
                G.GUST_BRASSEUR_ESTIM,
                G.GUST_BRASSEUR_LOBOU,
                G.GUST_BRASSEUR_UPBOU]
min_gust_levels = [0,5,10,20]
min_gust_levels = [5]
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
    data = calc_model_fields(data, i_model_fields)
    # join model and obs
    data = join_model_and_obs(data)
    # filter according to min gust strength
    data = EF.filter_according_obs_gust(data, min_gust)
    # join all model runs
    data = join_model_runs(data)
    # join stations
    data = join_all_stations(data)

    ###################### PLOT
    if i_plot > 0:
        df = data[G.BOTH][G.ALL_STAT]

        ymax = -np.Inf
        ymin = np.Inf
        xmax = -np.Inf
        xmin = np.Inf

        xs = {}
        ys = {}
        for mi,field_name in enumerate(i_model_fields):
            print(field_name)

            # prepare feature matrix
            if field_name == G.MODEL_MEAN_WIND:
                x = df[G.OBS_MEAN_WIND].values
                y = (df[field_name] - df[G.OBS_MEAN_WIND]).values
            else:
                x = df[G.OBS_GUST_SPEED].values
                y = (df[field_name] - df[G.OBS_GUST_SPEED]).values

            # delete NAN
            mask = np.isnan(x)
            mask[np.isnan(y)] = True
            x = x[~mask]
            y = y[~mask]

            # determine max/min y
            ymax = max(np.max(y),ymax)
            ymin = min(np.min(y),ymin)
            xmax = max(np.max(x),xmax)
            xmin = min(np.min(x),xmin)

            xs[field_name] = x
            ys[field_name] = y


        #xmin = min_gust
        #xmax = 50
        ymin = -30
        ymax = 50
        dx = 2
        dy = 2
        xseq = np.arange(np.floor(xmin),np.ceil(xmax),dx)
        yseq = np.arange(np.floor(ymin),np.ceil(ymax),dy)

        # Create Histrogram and find maximum occurence
        Hs = {}
        Hmax = -np.Inf
        for mi,field_name in enumerate(i_model_fields):
            H, xseq, yseq = np.histogram2d(xs[field_name], ys[field_name], bins=(xseq,yseq))
            H = np.sqrt(H)
            Hs[field_name] = H
            Hmax = max(np.max(H),Hmax)


        # plot preparation
        fig = plt.figure(figsize=(14,10))
        nrow = 2
        ncol = 3
        # loop over axes and gust calc method
        axes = []
        for mi,field_name in enumerate(i_model_fields):
            print(field_name)
            ax = fig.add_subplot(nrow, ncol, mi+1, xlim=xseq[[0,-1]], ylim=yseq[[0,-1]])
            axes.append(ax)

            norm = mpl.colors.Normalize(vmin=0, vmax=Hmax)
            im = mpl.image.NonUniformImage(ax, interpolation='bilinear', norm=norm)
            xcenters = (xseq[:-1] + xseq[1:])/2
            ycenters = (yseq[:-1] + yseq[1:])/2

            im.set_data(xcenters,ycenters, np.transpose(Hs[field_name]))
            ax.images.append(im)

            if field_name == G.MODEL_MEAN_WIND:
                xlab = 'Observed mean wind (OBS) [m/s]'
                ylab = 'Model mean wind error (MOD-OBS) [m/s]'
            else:
                xlab = 'Observed gust (OBS) [m/s]'
                ylab = 'Model gust error (MOD-OBS) [m/s]'

            ax.set_xlabel(xlab)
            if (mi == 0):
                ax.set_ylabel(ylab)
            ax.axhline(0, color='k', linewidth=0.8)
            ax.set_title(field_name)

        # finish plot
        title = CN.case_name + ' minGust ' + str(min_gust) +' m/s'
        plt.suptitle(title)
        plt.subplots_adjust(left=0.05,bottom=0.11,right=0.95,top=0.9,wspace=0.10,hspace=0.3)

        if i_plot == 1:
            plt.show()
        elif i_plot > 1:
            plot_name = CN.plot_path + 'scattercontour_minGust_'+str(min_gust).zfill(2)+'.png'
            print(plot_name)
            plt.savefig(plot_name)
            plt.close('all')



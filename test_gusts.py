import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from sklearn.linear_model import LinearRegression
from functions import calc_model_fields, join_model_and_obs, \
                        join_model_runs, join_all_stations, \
                        load_var_at_station_from_nc
import globals as G
from filter import EntryFilter
from namelist_cases import Case_Namelist

from netCDF4 import Dataset

############ USER INPUT #############

sel_stat = 'ABO'
#sel_stat = 'ATT'
#sel_stat = 'AGBAD'
#sel_stat = 'DLAUG'
#sel_stat = 'FRANY'
#sel_stat = 'MAS'
#sel_stat = 'OSRIN'
#sel_stat = 'SLFILI'
#sel_stat = 'TGKAL'
#sel_stat = 'AGAAR'

var_name = 'Z0'
var_name = 'T'
var_name = 'FF_10M'
var_name = 'VMAX_10M'
#var_name = 'TCM'
ndim = {
        'Z0':2,
        'T':3,
        'FF_10M':2,
        'VMAX_10M':2,
        'TCM':2
        }

## June pompa
#nc_path = '/scratch/heimc/wd/18063000_101/lm_coarse/'+var_name+'.nc'
#lm_run_string = '2018063000'
#case_index = 14
#
## June prerelease
#nc_path = '/scratch/heimc/wd/18062012_101/lm_coarse/'+var_name+'.nc'
#lm_run_string = '2018062012'
#case_index = 19

# Burglind pompa
nc_path = '/scratch/heimc/wd/18010300_101_pom/lm_coarse/'+var_name+'.nc'
lm_run_string = '2018010300'
case_index = 21

## Burglind prerelease (dp)
#nc_path = '/scratch/heimc/wd/18010300_101_pre_dp/lm_coarse/'+var_name+'.nc'
#lm_run_string = '2018010300'
#case_index = 20


# TEST OF FINAL OUTPUT
nc_path = '/scratch/heimc/wd/18010300_101/lm_coarse/'+var_name+'.nc'
lm_run_string = '2018010300'
case_index = 14


CN = Case_Namelist(case_index)
# do not plot (0) show plot (1) save plot (2)
i_plot = 0
# model fields to calculate 
i_model_fields = [G.GUST_MIX_COEF_LINEAR,
                  G.MODEL_MEAN_WIND,
                  G.MEAN_WIND_INST]
#i_model_fields = [G.GUST_BRASSEUR_ESTIM]
#####################################


# create directories
if i_plot > 1 and not os.path.exists(CN.plot_path):
    os.mkdir(CN.plot_path)

EF = EntryFilter()

# load data
data = pickle.load( open(CN.mod_path, 'rb') )


# LOAD STATION DATA
row_inds = (np.arange(1,25)*360).astype(np.int)-1
if var_name == 'Z0':
    var_stat = data[G.MODEL][G.STAT][sel_stat]['raw'][lm_run_string]['z0']
    var_stat = var_stat[row_inds]
elif var_name == 'T':
    var_stat = data[G.MODEL][G.STAT][sel_stat]['raw'][lm_run_string]['Tl1']
    var_stat = var_stat[row_inds]
elif var_name == 'FF_10M':
    ## from fortran file
    #var_stat = data[G.MODEL][G.STAT][sel_stat]['raw'][lm_run_string]['zvp10']
    #var_stat = var_stat[row_inds]
    # self calculated (nonesense but for testing
    data = calc_model_fields(data, i_model_fields)
    data = join_model_and_obs(data)
    data = join_model_runs(data)
    var_stat = data[G.MODEL][G.STAT][sel_stat]['fields'][lm_run_string][G.MEAN_WIND_INST]
elif var_name == 'VMAX_10M':
    data = calc_model_fields(data, i_model_fields)
    data = join_model_and_obs(data)
    data = join_model_runs(data)
    var_stat = data[G.MODEL][G.STAT][sel_stat]['fields'][lm_run_string][G.GUST_MIX_COEF_LINEAR]
    #var_stat = data[G.MODEL][G.STAT][sel_stat]['fields'][lm_run_string][G.GUST_BRASSEUR_ESTIM]
elif var_name == 'TCM':
    var_stat = data[G.MODEL][G.STAT][sel_stat]['raw'][lm_run_string]['tcm']
    var_stat = var_stat[row_inds]
else:
    raise NotImplementedError()


var_nc = load_var_at_station_from_nc(nc_path, var_name, sel_stat)

## GET STATION i AND j AND fort_stat INDEX
#station_file = "/users/heimc/stations/all_stations.lst"
#station_list = np.genfromtxt(station_file, skip_header=1, dtype=np.str)
#headers = station_list[0,:]
#station_list = station_list[1:,:]
##print(headers)
#sel_stat_ind = station_list[:,0] == sel_stat
#print(station_list[sel_stat_ind,:])
## final indices
#i_ind = station_list[sel_stat_ind,8].astype(np.int) - 1
#j_ind = station_list[sel_stat_ind,9].astype(np.int) - 1
#print(i_ind)
#print(j_ind)
#
#
## LOAD NC FILE DATA
#print('##############')
#
#ncf = Dataset(nc_path, 'r')
#if ndim[var_name] == 2:
#    all_var_nc = ncf[var_name]
#    var_nc = ncf[var_name][:,j_ind,i_ind].flatten()
#elif ndim[var_name] == 3:
#    kind = 79
#    all_var_nc = ncf[var_name][:,kind,:,:]
#    var_nc = ncf[var_name][:,kind,j_ind,i_ind].flatten()


# nice output for comparison
tmp = var_stat.to_frame()
tmp['nc'] = var_nc
try:
    tmp['abs error'] = tmp['nc'] - tmp[G.GUST_MIX_COEF_LINEAR]
except:
    pass
print(tmp)
#plt.plot(tmp)
#plt.show()
#quit()

if var_name == 'VMAX_10M':
    print('##### max abs error ####')
    abs_err = np.abs(var_nc - var_stat)
    print(np.max(abs_err))
    print('##### max rel error####')
    print(np.max(abs_err/var_nc))
else:
    print('##### max abs error ####')
    abs_err = np.abs(var_nc - var_stat)
    print(np.max(abs_err))
    print('##### max rel error ####')
    print(np.max(abs_err/var_nc))

quit()


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
            y = (df[field_name] - df[G.OBS_MEAN_WIND]).values
            xlab = 'Observed mean wind (OBS) [m/s]'
            ylab = 'Model mean wind error (MOD-OBS) [m/s]'
        else:
            X = df[G.OBS_GUST_SPEED].values.reshape(-1,1)
            y = (df[field_name] - df[G.OBS_GUST_SPEED]).values
            xlab = 'Observed gust (OBS) [m/s]'
            ylab = 'Model gust error (MOD-OBS) [m/s]'

        # delete NAN
        mask = np.isnan(X[:,0])
        mask[np.isnan(y)] = True
        X = X[~mask,:]
        y = y[~mask]


        # determine max/min y
        ymax = max(np.max(y),ymax)
        ymin = min(np.min(y),ymin)
        ymax = 70
        ymin = -50

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
        ax.scatter(X[:,0], y, color='black', marker=".")
        ax.plot(X, line, color='red')
        ax.plot(mp_x, mp_y, color='orange')
        ax.set_xlabel(xlab)
        if mi % ncol == 0 or field_name == G.MODEL_MEAN_WIND:
            ax.set_ylabel(ylab)
        ax.axhline(0, color='k', linewidth=0.8)
        ax.set_title(field_name)
    # set axes limits in each ax
    for i,ax in enumerate(axes):
        ax.set_ylim((ymin,ymax))
        ax.set_xlim(left=min_gust)
        if i == 0:
            ax.text(np.max(X)-0.13*(np.max(X)-np.min(X)), ymax-0.10*(ymax-ymin), len(y))

    # finish plot
    title = CN.case_name + ' minGust ' + str(min_gust) +' m/s'
    plt.suptitle(title)
    plt.subplots_adjust(left=0.05,bottom=0.08,right=0.95,top=0.9,wspace=0.2,hspace=0.3)

    if i_plot == 1:
        plt.show()
    elif i_plot > 1:
        if label == '':
            plot_name = CN.plot_path + 'scatter_minGust_'+str(min_gust).zfill(2)+'.png'
        else:
            plot_name = CN.plot_path + 'scatter_'+label+'_minGust_'+str(min_gust).zfill(2)+'.png'
        print(plot_name)
        plt.savefig(plot_name)
        plt.close('all')



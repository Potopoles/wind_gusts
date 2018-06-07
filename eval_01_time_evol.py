import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import copy
from functions import calc_model_fields
import globals as G


############ USER INPUT #############
# obs case name (name of obs pkl file in data folder)
obs_case_name = 'burglind'
#obs_case_name = 'foehn_apr18'
# model case name (name of folder with model data in 'mod_path'
model_case_name = 'burglind_ref'
#model_case_name = 'foehn_apr18_ref'
# obs model combination case name
obs_model_case_name = 'OBS_'+obs_case_name+'_MODEL_'+model_case_name
# mode of plotting: either ALL_G.STATIONS (1 plot for each station) or MEAN_OVER_G.STATIONS
plot_mode = 'ALL_STATIONS'
#plot_mode = 'MEAN_OVER_STATIONS'
#plot_mode = 'ABO'
# do not plot (0) show plot (1) save plot (2)
i_plot = 2
# gust methods to calculate
i_gust_fields = [G.GUST_MIX_COEF_LINEAR,
                G.GUST_MIX_COEF_NONLIN,
                G.GUST_BRASSEUR_ESTIM,
                G.GUST_BRASSEUR_LOBOU,
                G.GUST_BRASSEUR_UPBOU,
                G.KVAL_BRASSEUR_ESTIM,
                G.KVAL_BRASSEUR_LOBOU,
                G.KVAL_BRASSEUR_UPBOU]
# gust methods to plot on left axis
#i_gust_fields_plot = [G.GUST_MIX_COEF_LINEAR,
#                    G.GUST_MIX_COEF_NONLIN,
#                    G.GUST_BRASSEUR_ESTIM,
#                    G.GUST_BRASSEUR_LOBOU,
#                    G.GUST_BRASSEUR_UPBOU]
i_gust_fields_plot = [G.GUST_MIX_COEF_LINEAR,
                    G.GUST_MIX_COEF_NONLIN,
                    G.GUST_BRASSEUR_ESTIM]
# path of input obs_model pickle file
data_pickle_path = '../data/'+obs_model_case_name+'.pkl'
# directory to save plots
plot_base_dir = '../plots/'
plot_case_dir = plot_base_dir + obs_model_case_name + '/'
plot_out_dir = plot_case_dir + '/time_evolution/'
#####################################

# create directories
if i_plot > 1 and not os.path.exists(plot_case_dir):
    os.mkdir(plot_case_dir)
if i_plot > 1 and not os.path.exists(plot_out_dir):
    os.mkdir(plot_out_dir)

# load data
data = pickle.load( open(data_pickle_path, 'rb') )

# calculate gusts
data = calc_model_fields(data, i_gust_fields)

station_names = np.asarray(data[G.STAT_NAMES])
nstat = len(station_names)
nhrs = len(data[G.OBS][G.DTS])

# Calculate additional fields
mod_wind = np.full((nhrs,nstat),np.nan)

# Prepare index mask to map time step model output to hourly values
hr_inds = np.zeros((nhrs,360))
for i in range(0,nhrs):
    hr_inds[i,:] = i*360 + np.arange(0,360)
hr_inds = hr_inds.astype(np.int)

for si,stat in enumerate(station_names):

    for hr in range(0,nhrs):
        inds = hr_inds[hr]
        mod_wind[hr,si] = np.mean(data[G.MODEL][G.STAT][stat][G.PAR]['zvp10'][inds])


############################################################################
################################# PLOTTING #################################
############################################################################
def plot_station(stat):
    si = np.argwhere(station_names == stat).squeeze()
    lines = []
    labels = []

    dts = data[G.OBS][G.DTS]

    fig = plt.figure(figsize=(12,8))
    # model gusts
    for method in i_gust_fields_plot: 
        if method in [G.GUST_BRASSEUR_ESTIM, G.GUST_BRASSEUR_LOBOU, G.GUST_BRASSEUR_UPBOU]:
            line, = plt.plot(dts, data[G.MODEL][G.STAT][stat][G.FIELDS][method], color='red')
        else:
            line, = plt.plot(dts, data[G.MODEL][G.STAT][stat][G.FIELDS][method])
        lines.append(line)
        labels.append(method)

    ax = plt.gca()

    # brasseur gust range
    upper = data[G.MODEL][G.STAT][stat][G.FIELDS][G.GUST_BRASSEUR_UPBOU]
    lower = data[G.MODEL][G.STAT][stat][G.FIELDS][G.GUST_BRASSEUR_LOBOU]
    line = ax.fill_between(dts, lower, upper, where=upper >= lower, alpha=0.5)
    ax.fill_between(dts, lower, upper, where=upper < lower, alpha=0.5, color='orange')
    lines.append(line)
    labels.append('brass. bounds')

    # observed gust
    obs = data[G.OBS][G.STAT][stat][G.PAR]['VMAX_10M1']
    line, = plt.plot(dts, obs,color='black', linewidth=2)
    lines.append(line)
    labels.append('gust obs')

    # model mean wind
    line, = plt.plot(dts, mod_wind[:,si], linestyle='-.')
    lines.append(line)
    labels.append('wind model')

    # observed mean wind
    obs = data[G.OBS][G.STAT][stat][G.PAR]['FF_10M']
    line, = plt.plot(dts, obs, linestyle='-.')
    lines.append(line)
    labels.append('wind obs')


    ax.set_xlabel('time (UTC)')
    ax.set_ylabel('wind gust [m/s]')
    ax.set_title(stat)
    ax.grid()

    ax2 = ax.twinx()
    line, = ax2.plot(dts, data[G.MODEL][G.STAT][stat][G.FIELDS][G.KVAL_BRASSEUR_ESTIM],
                    linestyle='--', linewidth=1.0, color='grey')
    lines.append(line)
    labels.append('k bra est')
    line, = ax2.plot(dts, data[G.MODEL][G.STAT][stat][G.FIELDS][G.KVAL_BRASSEUR_LOBOU],
                    linestyle='--', linewidth=1.0, color='black')
    lines.append(line)
    labels.append('k bra lobou')
    line, = ax2.plot(dts, data[G.MODEL][G.STAT][stat][G.FIELDS][G.KVAL_BRASSEUR_UPBOU],
                    linestyle='--', linewidth=1.0, color='brown')
    lines.append(line)
    labels.append('k bra upbou')


    ax2.set_ylabel('brasseur model level')
    ax2.set_ylim((0,80))

    plt.legend(lines,labels)

    if i_plot == 1:
        plt.show()
    elif i_plot > 1:
        plot_name = plot_out_dir + stat + '.png'
        plt.savefig(plot_name)
        plt.close('all')




if plot_mode == 'ALL_STATIONS':
    for stat in station_names[0:150]:
        print(stat)
        plot_station(stat)


elif plot_mode == 'MEAN_OVER_STATIONS':
    mean_stat_name = 'MEAN_OVER_STATIONS'

    # copy from any station
    data[G.OBS][G.STAT][mean_stat_name] = copy.deepcopy(data[G.OBS][G.STAT][station_names[0]])
    data[G.MODEL][G.STAT][mean_stat_name] = copy.deepcopy(data[G.MODEL][G.STAT][station_names[0]])

    # model variables
    for method in i_gust_fields:
        vals_all_stat = np.full((nhrs,nstat),np.nan)
        for si,stat in enumerate(station_names):
            vals_all_stat[:,si] = data[G.MODEL][G.STAT][stat][G.FIELDS][method]
        vals_mean_stat = np.mean(vals_all_stat,axis=1)
        data[G.MODEL][G.STAT][mean_stat_name][G.FIELDS][method] = vals_mean_stat

    # obs variables
    obs_names = ['VMAX_10M1', 'FF_10M']
    for obs_name in obs_names:
        for si,stat in enumerate(station_names):
            vals_all_stat[:,si] = data[G.OBS][G.STAT][stat][G.PAR][obs_name]
        vals_mean_stat = np.nanmean(vals_all_stat,axis=1)
        data[G.OBS][G.STAT][mean_stat_name][G.PAR][obs_name] = vals_mean_stat


    station_names = np.append(station_names, mean_stat_name)

    # model mean wind
    si = np.argwhere(station_names == mean_stat_name).squeeze()
    mod_wind = np.insert(mod_wind, si, np.mean(mod_wind, axis=1), axis=1) 
    plot_station(mean_stat_name)


# plot a single station
else:
    stat = plot_mode
    if stat in station_names:
        print('station ' + stat)
        plot_station(stat)
    else:
        raise ValueError('Unknown station name or option given by user for variable "plot_mode"!') 




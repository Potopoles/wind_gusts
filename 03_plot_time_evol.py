import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from functions import calc_gusts
import globals as G


############ USER INPUT #############
# obs case name (name of obs pkl file in data folder)
obs_case_name = 'burglind'
# model case name (name of folder with model data in 'mod_path'
model_case_name = 'burglind_ref'
# obs model combination case name
obs_model_case_name = 'G.OBS_'+obs_case_name+'_G.MODEL_'+model_case_name
# mode of plotting: either ALL_G.STATIONS (1 plot for each station) or MEAN_OVER_G.STATIONS
plot_mode = 'ALL_STATIONS'
#plot_mode = 'MEAN_OVER_STATIONS'
#plot_mode = 'SIO'
# do not plot (0) show plot (1) save plot (2)
i_plot = 1
# gust methods to calculate
i_gust_fields = [G.GUST_MIX_COEF_LINEAR,
                G.GUST_MIX_COEF_NONLIN,
                G.GUST_BRASSEUR_ESTIM,
                G.GUST_BRASSEUR_K_VAL]
# gust methods to plot on left axis
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
data = calc_gusts(data, i_gust_fields)

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
        line, = plt.plot(dts, data[G.MODEL][G.STAT][stat][G.GUST][method])
        lines.append(line)
        labels.append(method)
    # observed gust
    obs = data[G.OBS][G.STAT][stat][G.PAR]['VMAX_10M1']
    line, = plt.plot(dts, obs,color='black')
    lines.append(line)
    labels.append('gust obs')
    # model mean wind
    line, = plt.plot(dts, mod_wind[:,si])
    lines.append(line)
    labels.append('wind model')
    # observed mean wind
    obs = data[G.OBS][G.STAT][stat][G.PAR]['FF_10M']
    line, = plt.plot(dts, obs)
    lines.append(line)
    labels.append('wind obs')

    ax = plt.gca()

    ax.set_xlabel('simulation hour')
    ax.set_ylabel('wind/gust [m/s]')
    ax.set_title(stat)
    ax.grid()

    ax2 = ax.twinx()
    line, = ax2.plot(dts, data[G.MODEL][G.STAT][stat][G.GUST][G.GUST_BRASSEUR_K_VAL],
                    linestyle='-', linewidth=1.0, color='grey')
    lines.append(line)
    labels.append('k bra')

    ax2.set_ylabel('brasseur model level')
    ax2.set_ylim((20,80))

    plt.legend(lines,labels)

    if i_plot == 1:
        plt.show()
    elif i_plot > 1:
        plot_name = plot_out_dir + stat + '.png'
        plt.savefig(plot_name)
        plt.close('all')




if plot_mode == 'ALL_STATIONS':
    for stat in station_names:
        print(stat)
        plot_station(stat)


# plot a single station
else:
    stat = plot_mode
    if stat in station_names:
        print('station ' + stat)
        plot_station(stat)
    else:
        raise ValueError('Unknown station name or option given by user for variable "plot_mode"!') 



#elif plot_mode == 'MEAN_OVER_STATIONS':
#    mod_gust_stat_mean = np.nanmean(mod_gust, axis=1)
#    obs_gust_stat_mean = np.nanmean(obs_gust, axis=1)
#    mod_wind_stat_mean = np.nanmean(mod_wind, axis=1)
#    obs_wind_stat_mean = np.nanmean(obs_wind, axis=1)
#    hypot_gust_stat_mean = np.nanmean(hypot_gust, axis=1)
#
#    lines = []
#    labels = []
#    # model gusts
#    for i in range(0,4):
#        if i+1 in i_methods:
#            line, = plt.plot(mod_gust_stat_mean[:,i])
#            lines.append(line)
#            labels.append('gust method '+str(i+1))
#    # observed gust
#    line, = plt.plot(obs_gust_stat_mean,color='black')
#    lines.append(line)
#    labels.append('gust obs')
#    # model mean wind
#    line, = plt.plot(mod_wind_stat_mean)
#    lines.append(line)
#    labels.append('wind model')
#    # observed mean wind
#    line, = plt.plot(obs_wind_stat_mean)
#    lines.append(line)
#    labels.append('wind obs')
#    # hypothetical model gust method 1
#    # based on observed wind
#    line, = plt.plot(hypot_gust_stat_mean, linestyle='--')
#    lines.append(line)
#    labels.append('hypoth. gust')
#
#    plt.legend(lines,labels)
#    
#    ax = plt.gca()
#
#    ax.set_xlabel('simulation hour')
#    ax.set_ylabel('wind/gust ['+unit+']')
#    ax.set_title(stat)
#
#
#    if i_plot == 1:
#        plt.show()
#    elif i_plot > 1:
#        plot_name = plot_out_dir + 'all' + '.png'
#        plt.savefig(plot_name)
#        plt.close('all')


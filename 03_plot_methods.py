import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle


############ USER INPUT #############
# obs case name (name of obs pkl file in data folder)
obs_case_name = 'burglind'
# model case name (name of folder with model data in 'mod_path'
model_case_name = 'burglind_ref'
# mode of plotting: either ALL_STATIONS (1 plot for each station) or MEAN_OVER_STATIONS
#plot_mode = 'ALL_STATIONS'
#plot_mode = 'MEAN_OVER_STATIONS'
plot_mode = 'ABO'
# save output (1) or plot output (0)
i_save = 0 
# gust methods to calculate and plot
# i_method = 1: estimate from zvp10 and ustar
# i_method = 2: estimate from zvp30 and ustar
# i_method = 3: brasseur
# i_method = 4: estimate from zvp10 and ustar and gust factor
i_methods = [1,3,4]
# path of input obs_model pickle file
data_pickle_path = '../data/OBS_'+obs_case_name+'_MODEL_'+model_case_name+'.pkl'
# directory to save plots
plot_dir = '../plots/'
MODEL = 'model'
OBS = 'obs'
unit = 'km/h'
#unit = 'm/s'
#####################################

data = pickle.load( open(data_pickle_path, 'rb') )

station_names = data['station_names']
nstat = len(station_names)
nmethods = len(i_methods)

# Prepare index mask to map model output to observation values
nhrs = len(data['obs_dts'])
hr_inds = np.zeros((nhrs,360))
for i in range(0,nhrs):
    hr_inds[i,:] = i*360 + np.arange(0,360)
hr_inds = hr_inds.astype(np.int)

#print(data[MODEL]['ABO']['tcm'].shape[0]/360)
#quit()

# value fields
mod_gust = np.full((nhrs,nstat,4),np.nan)
obs_gust = np.full((nhrs,nstat),np.nan)
mod_wind = np.full((nhrs,nstat),np.nan)
obs_wind = np.full((nhrs,nstat),np.nan)
# hypothetical gust that model method 1 calculates from observed wind
hypot_gust = np.full((nhrs,nstat),np.nan)
# error fields
mod_err_gust = np.full((nhrs,nstat,4),np.nan)
abs_err_gust = np.full((nhrs,nstat,4),np.nan)
# k bra
k_bra = np.full((nhrs,nstat),np.nan)


# loop through all stations
for si,stat in enumerate(station_names):

    # unit conversion
    if unit == 'km/h':
        data[MODEL][stat]['zvp10'] = data[MODEL][stat]['zvp10']*3.6
        data[MODEL][stat]['zvp30'] = data[MODEL][stat]['zvp30']*3.6
        data[MODEL][stat]['zvpb'] = data[MODEL][stat]['zvpb']*3.6
        data[MODEL][stat]['zv_bra'] = data[MODEL][stat]['zv_bra']*3.6
        data[OBS][stat]['VMAX_10M1'] = data[OBS][stat]['VMAX_10M1']*3.6
        data[OBS][stat]['FF_10M'] = data[OBS][stat]['FF_10M']*3.6

    for hr in range(0,nhrs):
        inds = hr_inds[hr]
        obs_gust[hr,si] = data[OBS][stat]['VMAX_10M1'][hr] 
        mod_wind[hr,si] = np.mean(data[MODEL][stat]['zvp10'][inds])
        obs_wind[hr,si] = data[OBS][stat]['FF_10M'][hr] 
        k_bra[hr,si] = np.mean(data[MODEL][stat]['k_bra'][inds])

    for mi in i_methods:


        if mi == 1:

            tcm = data[MODEL][stat]['tcm']
            zcm = tcm
            zcm[zcm < 5E-4] = 5E-4
            zsqcm = np.sqrt(zcm)
            zvp10 = data[MODEL][stat]['zvp10']
            gust = zvp10 + 3.0 * 2.4 * zsqcm * zvp10

            # hypothetical observation based gust
            hyp_gust_fact = 3.0 * 2.4 * zsqcm 
            for hr in range(0,nhrs):
                inds = hr_inds[hr]
                hr_max_hyp_gust_fact = np.max(hyp_gust_fact[inds])
                hyp_gust = obs_wind[hr,si] + hr_max_hyp_gust_fact * obs_wind[hr,si]
                hypot_gust[hr,si] = hyp_gust


        if mi == 2:

            tcm = data[MODEL][stat]['tcm']
            zcm = tcm
            zcm[zcm < 5E-4] = 5E-4
            zsqcm = np.sqrt(zcm)
            zvp30 = data[MODEL][stat]['zvp30']
            zvpb = data[MODEL][stat]['zvpb']
            gust = zvp30 + 3.0 * 2.4 * zsqcm * zvpb

        if mi == 3:

            gust = data[MODEL][stat]['zv_bra']

        if mi == 4:

            tcm = data[MODEL][stat]['tcm']
            zcm = tcm
            zcm[zcm < 5E-4] = 5E-4
            zsqcm = np.sqrt(zcm)
            zvp10 = data[MODEL][stat]['zvp10']
            if unit == 'km/h':
                gust = zvp10 + (3.0 * 2.4 + 0.09 * zvp10/3.6) * zsqcm * zvp10
            elif unit == 'm/s':
                gust = zvp10 + (3.0 * 2.4 + 0.09 * zvp10) * zsqcm * zvp10
    
        for hr in range(0,nhrs):
            inds = hr_inds[hr]
            # find maximum gust
            hr_max_gust = np.max(gust[inds])

            mod_gust[hr,si,mi-1] = hr_max_gust



for mi in i_methods:
    mod_err_gust[:,:,mi-1] = mod_gust[:,:,mi-1] - obs_gust
abs_err_gust = np.abs(mod_err_gust)





if plot_mode == 'ALL_STATIONS':
    for si,stat in enumerate(station_names):
        print('station ' + stat)
        lines = []
        labels = []

        fig = plt.figure(figsize=(12,8))
        # model gusts
        for i in range(0,4):
            if i+1 in i_methods:
                line, = plt.plot(mod_gust[:,si,i])
                lines.append(line)
                labels.append('gust method '+str(i+1))
        # observed gust
        line, = plt.plot(obs_gust[:,si],color='black')
        lines.append(line)
        labels.append('gust obs')
        # model mean wind
        line, = plt.plot(mod_wind[:,si])
        lines.append(line)
        labels.append('wind model')
        # observed mean wind
        line, = plt.plot(obs_wind[:,si])
        lines.append(line)
        labels.append('wind obs')
        # hypothetical model gust method 1
        # based on observed wind
        line, = plt.plot(hypot_gust[:,si], linestyle='--')
        lines.append(line)
        labels.append('hypoth. gust')

        ax = plt.gca()

        ax.set_xlabel('simulation hour')
        ax.set_ylabel('wind/gust ['+unit+']')
        ax.set_title(stat)
        ax.grid()

        ax2 = ax.twinx()
        line, = ax2.plot(k_bra[:,si], linestyle='-', linewidth=1.0, color='grey')
        lines.append(line)
        labels.append('k bra')

        ax2.set_ylabel('brasseur model level')
        ax2.set_ylim((20,80))

        plt.legend(lines,labels)

        plot_name = plot_dir + stat + '.png'

        if i_save == 0:
            plt.show()
        else:
            plt.savefig(plot_name)
            plt.close('all')






elif plot_mode == 'MEAN_OVER_STATIONS':
    mod_gust_stat_mean = np.nanmean(mod_gust, axis=1)
    obs_gust_stat_mean = np.nanmean(obs_gust, axis=1)
    mod_wind_stat_mean = np.nanmean(mod_wind, axis=1)
    obs_wind_stat_mean = np.nanmean(obs_wind, axis=1)
    hypot_gust_stat_mean = np.nanmean(hypot_gust, axis=1)

    lines = []
    labels = []
    # model gusts
    for i in range(0,4):
        if i+1 in i_methods:
            line, = plt.plot(mod_gust_stat_mean[:,i])
            lines.append(line)
            labels.append('gust method '+str(i+1))
    # observed gust
    line, = plt.plot(obs_gust_stat_mean,color='black')
    lines.append(line)
    labels.append('gust obs')
    # model mean wind
    line, = plt.plot(mod_wind_stat_mean)
    lines.append(line)
    labels.append('wind model')
    # observed mean wind
    line, = plt.plot(obs_wind_stat_mean)
    lines.append(line)
    labels.append('wind obs')
    # hypothetical model gust method 1
    # based on observed wind
    line, = plt.plot(hypot_gust_stat_mean, linestyle='--')
    lines.append(line)
    labels.append('hypoth. gust')

    plt.legend(lines,labels)
    
    ax = plt.gca()

    ax.set_xlabel('simulation hour')
    ax.set_ylabel('wind/gust ['+unit+']')
    ax.set_title(stat)

    plot_name = plot_dir + 'all' + '.png'

    if i_save == 0:
        plt.show()
    else:
        plt.savefig(plot_name)
        plt.close('all')



# plot a single station
else:
    stat = plot_mode
    if stat in station_names:
        print('station ' + stat)
        lines = []
        labels = []

        fig = plt.figure(figsize=(12,8))
        # model gusts
        for i in range(0,4):
            if i+1 in i_methods:
                line, = plt.plot(mod_gust[:,si,i])
                lines.append(line)
                labels.append('gust method '+str(i+1))
        # observed gust
        line, = plt.plot(obs_gust[:,si],color='black')
        lines.append(line)
        labels.append('gust obs')
        # model mean wind
        line, = plt.plot(mod_wind[:,si])
        lines.append(line)
        labels.append('wind model')
        # observed mean wind
        line, = plt.plot(obs_wind[:,si])
        lines.append(line)
        labels.append('wind obs')
        # hypothetical model gust method 1
        # based on observed wind
        line, = plt.plot(hypot_gust[:,si], linestyle='--')
        lines.append(line)
        labels.append('hypoth. gust')

        ax = plt.gca()

        ax.set_xlabel('simulation hour')
        ax.set_ylabel('wind/gust ['+unit+']')
        ax.set_title(stat)
        ax.grid()

        ax2 = ax.twinx()
        line, = ax2.plot(k_bra[:,si], linestyle='-', linewidth=1.0, color='grey')
        lines.append(line)
        labels.append('k bra')

        ax2.set_ylabel('brasseur model level')
        ax2.set_ylim((20,80))

        plt.legend(lines,labels)

        plot_name = plot_dir + stat + '.png'

        if i_save == 0:
            plt.show()
        else:
            plt.savefig(plot_name)
            plt.close('all')
    else:
        raise ValueError('Unknown station name or option given by user for variable "plot_mode"!') 

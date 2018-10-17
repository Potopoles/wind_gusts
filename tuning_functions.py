import numpy as np
#import pandas as pd
import globals as G
#from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial import KDTree
#from netCDF4 import Dataset
#import time
from datetime import datetime, timedelta


#def apply_full_scaling(var):
#    mean_var = np.mean(var)
#    var = var - mean_var
#    sd_var = np.std(var)
#    var = var/sd_var
#    return(var, mean_var, sd_var)


def apply_scaling(var):
    sd_var = np.std(var)
    var = var/sd_var
    return(var, sd_var)




def prepare_model_params(use_model_fields, lm_runs, CN,
                        stat_fort_ind, stat_key, mod_stations,
                        model_params, model_dt, nhrs_forecast,
                        data_obs_stat, progress, njobs):

    # user output
    #if njobs > 1:
    #    progress += 1
    print('############# use ' + stat_key + '\t' + \
            str(round(progress/len(mod_stations),2)))


    # find indices for columns in model time step files
    tstep_col_ind = np.argwhere(np.asarray(model_params) == 'tstep')[0]
    zvp10_col_ind = np.argwhere(np.asarray(model_params) == 'zvp10')[0]

    # prepare station observation fields
    n_hours_all_lm = len(lm_runs)*nhrs_forecast
    obs_mean_stat  = np.full( ( n_hours_all_lm ), np.nan )
    obs_gust_stat  = np.full( ( n_hours_all_lm ), np.nan )

    # prepare station model_fields
    model_fields_stat = {}
    for field_name in use_model_fields:
        model_fields_stat[field_name] = \
                np.full( ( n_hours_all_lm, int(3600/model_dt) ), np.nan )

    ## add model data
    #raw_data = {}
    ##raw_data2 = {}
    #stat_data = {G.RAW:raw_data}
    ##stat_data = {G.RAW:raw_data, G.RAW2:raw_data2}
    #data[G.MODEL][G.STAT][stat_key] = stat_data

    # load data from all lm runs
    for lmI,lm_run in enumerate(lm_runs):

        ###################################################################
        ####### PART 1: Read and format model time step output file
        ###################################################################
        # construct file path
        mod_file_path = CN.raw_mod_path + lm_run + '/' + 'fort.' + str(stat_fort_ind)

        # time of first time step
        start_time = datetime.strptime(lm_run, '%Y%m%d%H')

        # opt 1: ignore invalid values but warn
        #values = np.genfromtxt(mod_file_path, delimiter=',',\
        #           dtype=np.float, loose=1)
        #if np.sum(np.isnan(values)) > 0:
        #    print(str(np.sum(np.isnan(values))) + ' invalid values!')
        # opt 2: raise error for invalid values
        values = np.genfromtxt(mod_file_path, delimiter=',',\
                    dtype=np.float)

        # After read in time step 0 has to be removed.
        # However, each time step has n_grid_points file rows,
        # where n_grid_points is the number
        # of output grid points around each station.
        # Therefore, first determine n_grid_points and remove the
        # n_grid_points first lines of file.
        n_grid_points = 0
        n_grid_points = np.sum(values[:,tstep_col_ind] == 0)
        if n_grid_points == 0:
            raise ValueError('n_grid_points is 0! Possibly ' + \
                                'model output is not as expected')
        values = values[n_grid_points:,:]
        time_steps = np.unique(values[:,tstep_col_ind])
        n_time_steps = len(time_steps)
        if not values.shape[0]/n_time_steps == n_grid_points:
            raise ValueError('number of entries does not ' + \
                                'divide by n_time_steps!')


        ###################################################################
        ####### PART 2: For each hour find best fitting model grid point
        ###################################################################
        # 1) For each hour, find for all of the n_grid_points of model 
        #    output the corresponding 3600/dt time steps
        # 2) Calculate the model hourly mean wind. Compare to observed
        #    hourly mean wind and determine the best fitting model grid
        #    point.
        # 3) Only keep this one and store in final_values

        nts_per_hr = int(3600/model_dt)
        #final_values = np.full( (n_hours_all_lm*nts_per_hr, values.shape[1]), np.nan )

        # TODO remove this after testing
        # contains for each hour the index of the best fitting model
        # grid point
        best_fit_gp_inds = np.full(nhrs_forecast, np.nan)
        best_mean_winds = np.full(nhrs_forecast, np.nan)

        # loop over hours in lm_run
        # hour label corresponds to time before label
        hrs = range(1,nhrs_forecast+1)
        for hI,hr in enumerate(hrs):
            #hr = 15

            # contains model mean wind values for each grid point and
            # time step of given hour
            model_mean_gp = np.zeros( (n_grid_points, nts_per_hr) )

            # loop over time steps of current hour
            # attention: this happens in model counter style starting @ 1)
            #            not in python style (starting @ 0))
            ts_inds = range((hr-1)*nts_per_hr+1, hr*nts_per_hr+1)
            for tsI,ts_ind in enumerate(ts_inds):
                row_inds = range((ts_ind-1)*n_grid_points, \
                                ts_ind*n_grid_points)
                model_mean_gp[:,tsI] = values[row_inds,zvp10_col_ind]
            model_mean_wind = np.mean(model_mean_gp, axis=1)

            # get observation date
            cur_time = start_time + timedelta(hours=hr)
            obs_mean_hr  = data_obs_stat[G.OBS_MEAN_WIND].\
                                loc[cur_time]
            obs_gust_hr = data_obs_stat[G.OBS_GUST_SPEED].\
                                loc[cur_time]

            # select best fitting model grid point
            grid_point_ind = np.argmin(np.abs(
                                model_mean_wind - obs_mean_hr))
            # TODO testing
            best_fit_gp_inds[hI] = grid_point_ind
            best_mean_winds[hI] = model_mean_wind[grid_point_ind]

            # select best fitting rows with time steps of best fitting
            # grid point
            sel_inds = (np.asarray(ts_inds)-1) * n_grid_points + \
                                                grid_point_ind
            #sel_values = values[sel_inds,:]

            hr_sel_inds = lmI*nhrs_forecast + hI
            obs_mean_stat[hr_sel_inds] = obs_mean_hr
            obs_gust_stat[hr_sel_inds] = obs_gust_hr

            for field_name in use_model_fields:
                col_ind = np.argwhere(
                            np.asarray(model_params) == field_name)[0]
                model_fields_stat[field_name][hr_sel_inds,:] = \
                                        values[sel_inds,col_ind]
                
            #zvp10_loc   [hr_sel_inds,:] = values[sel_inds,zvp10_col_ind]
            #tcm_loc     [hr_sel_inds,:] = values[sel_inds,tcm_col_ind]
            #print('nan ' + str(np.sum(np.isnan(zvp10))))


            ## store selection of current hour in final value array for 
            ## this lm_run
            #final_values[hI*nts_per_hr:(hI+1)*nts_per_hr, :] = sel_values


        ## Save in raw_data dictionary
        #ts_secs = (np.arange(1,nhrs_forecast*nts_per_hr+1)*model_dt).\
        #            astype(np.float)
        #dts = [start_time + timedelta(seconds=ts_sec) for ts_sec in ts_secs]
        #df = pd.DataFrame(final_values, index=dts, columns=model_params)
        #raw_data[lm_run] = df

    #print('obs mean wind')
    #print(obs_mean[:,sI])
    #print('best model mean wind')
    #print(np.round(best_mean_winds,1))



    result = (  obs_mean_stat,
                obs_gust_stat,
                model_fields_stat
             )
    return(result)



# THIS IS WAY TOO SLOW
#def get_point_col(vals1, vals2):
#    print('start')
#    t0 = time.time()
#    max_rad = 1.0 
#
#    cmap = plt.cm.get_cmap('gray')
#
#    N = len(vals1)
#    nk_array = np.zeros( (N, 2) )
#    nk_array[:,0] = vals1
#    nk_array[:,1] = vals2
#
#    kdtree = KDTree(nk_array, leafsize=10)
#    result = kdtree.query_ball_tree(kdtree, r=max_rad, eps=1.3)
#    result = np.asarray([len(res) for res in result])
#    #result = 1 - np.sqrt(result/np.max(result))
#    #result = 1 - result/np.max(result)
#    result = np.sqrt(result/np.max(result))
#    col = cmap(result)
#    t1 = time.time()
#    print(t1-t0)
#    return(col)

# THIS IS NICE!
def get_point_col(vals1, vals2, xmin=0, xmax=100, ymin=-100, ymax=100):
    #dx = 0.5
    #dy = 1
    dx = 1
    dy = 1
    dx = dx*1
    dy = dy*1
    xseq = np.arange(np.floor(xmin),np.ceil(xmax),dx)
    yseq = np.arange(np.floor(ymin),np.ceil(ymax),dy)
    H, xseq, yseq = np.histogram2d(vals1, vals2, bins=(xseq,yseq), normed=True)
    inds1 = np.digitize(vals1, xseq) - 1
    inds2 = np.digitize(vals2, yseq) - 1
    result = H[inds1,inds2]

    cmap = plt.cm.get_cmap('gray')

    result = np.sqrt(np.sqrt(result/np.max(result)))
    col = cmap(result)
    return(col)



def rotate(xy, angle):
    rot_mat = np.column_stack( [[ np.cos(angle), np.sin(angle)],
                                [-np.sin(angle), np.cos(angle)]] )
    xy_rot = rot_mat @ xy
    return(xy_rot)


def draw_error_percentile_lines(x, y, ax):

    # PREPARE PERCENTILE LINES
    ###############
    q0 = 10
    q1 = 50
    q2 = 90
    qs = [q0,q1,q2]
    qscol = ['red','orange','red']
    qslst = ['--','-','--']
    qslwd = [1.5,2,1.5]
    dmp = 2. # = dx
    ###############

    xy = np.row_stack( [x, y] )
    xy_rot = rotate(xy, -np.pi/4)

    speed = xy_rot[0,:]
    error = xy_rot[1,:]

    #ax.scatter(speed, error, color='grey', marker=".")

    mp_borders = np.arange(np.floor(0),np.ceil(np.max(speed)),dmp)
    mp_x = mp_borders[:-1] + np.diff(mp_borders)/2
    mgog = np.full((len(mp_x),3),np.nan)
    for qi in range(0,len(qs)):
        for i in range(0,len(mp_x)):
            # model gust vs obs gust
            inds = (speed > mp_borders[i]) & (speed <= mp_borders[i+1])
            if np.sum(inds) > 10:
                mgog[i,qi] = np.percentile(error[inds],q=qs[qi])

    for qi in range(0,len(qs)):
        #ax.plot(mp_x, mgog[:,qi], color=qscol[qi], linestyle=qslst[qi], linewidth=qslwd[qi])

        # backtransform percentile lines
        line_rot = np.row_stack( [mp_x, mgog[:,qi]] )
        line = rotate(line_rot, np.pi/4)
        ax.plot(line[0,:], line[1,:], color=qscol[qi], linestyle=qslst[qi], linewidth=qslwd[qi])

def draw_error_grid(xmax, ymax, ax):
    ax.plot([0,xmax], [0,ymax], color='k', linewidth=0.5)
    dy = 10/np.sqrt(2)*2
    for i in range(-5,6):
        if i is not 0:
            ax.plot([0,xmax], [i*dy,ymax+i*dy], color='grey', linewidth=0.5)


def draw_scatterplot(xvals, yvals, xlims, ylims,
                    xlab, ylab, title, ax):
    ax.scatter(xvals, yvals, color=get_point_col(xvals, yvals), marker=".")
    draw_error_grid(xlims[1], ylims[1], ax)
    draw_error_percentile_lines(xvals, yvals, ax)
    ax.axhline(y=0,c='grey')
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

def plot_mod_vs_obs(obs, gust, gust_init, obs_mean, mod_mean):

    xlims = (0,60)
    ylims = (0,60)

    xlims_mean = (0,40)
    ylims_mean = (0,40)

    fig,axes = plt.subplots(1,3, figsize=(18,5.3))

    ##########################################################################
    # obs gust vs model gust initial
    xlab = 'MOD gust [m/s]'
    ylab = 'OBS gust [m/s]'
    title = 'original MOD gust vs OBS gust'
    ax = axes[0]
    draw_scatterplot(gust_init, obs, xlims, ylims,
                    xlab, ylab, title, ax)

    ##########################################################################
    # obs gust vs model gust
    xlab = 'MOD gust [m/s]'
    ylab = 'OBS gust [m/s]'
    title = 'MOD gust vs OBS gust'
    ax = axes[1]
    draw_scatterplot(gust, obs, xlims, ylims,
                    xlab, ylab, title, ax)


    ##########################################################################
    # obs mean vs model mean
    xlab = 'MOD mean wind [m/s]'
    ylab = 'OBS mean wind [m/s]'
    title = 'MOD wind vs OBS wind'
    ax = axes[2]
    draw_scatterplot(mod_mean, obs_mean, xlims_mean, ylims_mean,
                    xlab, ylab, title, ax)


def plot_error(obs, model_mean, obs_mean, gust, gust_init):

    i_fancy_colors = 1

    fig,axes = plt.subplots(2,5, figsize=(19,8))


    xmin = 0
    xmax = 60
    ymin = -50
    ymax = 50

    q0 = 10
    q1 = 50
    q2 = 90
    qs = [q0,q1,q2]
    qscol = ['red','orange','red']
    qslst = ['--','-','--']
    qslwd = [1,2,1]

    dmp = 1. # = dx
    dy = 1.
    

    err = gust - obs
    err_init = gust_init - obs
    err_mean = model_mean - obs_mean
    rmse = np.sqrt(np.sum(err**2)/len(err))
    rmse_init = np.sqrt(np.sum(err_init**2)/len(err_init))
    err_mean_nonan = err_mean[~np.isnan(err_mean)] 
    rmse_mean = np.sqrt(np.sum(err_mean_nonan**2)/len(err_mean_nonan))
    print('RMSE mean wind: ' + str(rmse_mean))

    N = len(err)

    col_amplify = 6

    # calculate median
    mp_borders = np.arange(np.floor(xmin),np.ceil(xmax),dmp)
    mp_x = mp_borders[:-1] + np.diff(mp_borders)/2
    mp_borders_me = np.arange(np.floor(ymin),np.ceil(ymax),dmp)
    mp_x_me = mp_borders_me[:-1] + np.diff(mp_borders_me)/2
    mp_y_og = np.full((len(mp_x),3),np.nan)
    mp_y_init_og = np.full((len(mp_x),3),np.nan)
    mp_y_mg = np.full((len(mp_x),3),np.nan)
    mp_y_mg_init = np.full((len(mp_x),3),np.nan)
    mp_y_mm = np.full((len(mp_x),3),np.nan)
    mp_y_mm_init = np.full((len(mp_x),3),np.nan)
    mp_y_om = np.full((len(mp_x),3),np.nan)
    mp_y_om_init = np.full((len(mp_x),3),np.nan)
    mp_y_mean = np.full((len(mp_x),3),np.nan)
    mp_y_gg = np.full((len(mp_x),3),np.nan)
    mp_y_geme = np.full((len(mp_x_me),3),np.nan)

    for qi in range(0,len(qs)):
        for i in range(0,len(mp_x)):
            # model gust err vs obs gust
            inds = (obs > mp_borders[i]) & (obs <= mp_borders[i+1])
            if np.sum(inds) > 10:
                mp_y_og[i,qi] = np.percentile(err[inds],q=qs[qi])
                mp_y_init_og[i,qi] = np.percentile(err_init[inds],q=qs[qi])

            # model gust err vs model gust
            inds = (gust > mp_borders[i]) & (gust <= mp_borders[i+1])
            if np.sum(inds) > 10:
                mp_y_mg[i,qi] = np.percentile(err[inds],q=qs[qi])

            inds = (gust_init > mp_borders[i]) & (gust_init <= mp_borders[i+1])
            if np.sum(inds) > 10:
                mp_y_mg_init[i,qi] = np.percentile(err_init[inds],q=qs[qi])

            # model gust err vs model mean
            inds = (model_mean > mp_borders[i]) & (model_mean <= mp_borders[i+1])
            if np.sum(inds) > 10:
                mp_y_mm[i,qi] = np.percentile(err[inds],q=qs[qi])
                mp_y_mm_init[i,qi] = np.percentile(err_init[inds],q=qs[qi])

            # model gust err vs obs mean
            inds = (obs_mean > mp_borders[i]) & (obs_mean <= mp_borders[i+1])
            if np.sum(inds) > 10:
                mp_y_om[i,qi] = np.percentile(err[inds],q=qs[qi])
                mp_y_om_init[i,qi] = np.percentile(err_init[inds],q=qs[qi])

            # model mean err vs model mean
            #inds = (model_mean > mp_borders[i]) & (model_mean <= mp_borders[i+1])
            inds = (obs_mean > mp_borders[i]) & (obs_mean <= mp_borders[i+1])
            if np.sum(inds) > 10:
                mp_y_mean[i,qi] = np.nanpercentile(err_mean[inds],q=qs[qi])
            ## obs gust vs model gust
            #inds = (gust > mp_borders[i]) & (gust <= mp_borders[i+1])
            #if np.sum(inds) > 10:
            #    mp_y_gg[i,qi] = np.percentile(obs[inds],q=qs[qi])

            # model gust err vs model mean err
            inds = (err_mean > mp_borders_me[i]) & (err_mean <= mp_borders_me[i+1])
            if np.sum(inds) > 10:
                mp_y_geme[i,qi] = np.percentile(err[inds],q=qs[qi])
                #mp_y_geme[i,qi] = np.percentile(err_init[inds],q=qs[qi])


    xlab = 'Observed gust (OBS) [m/s]'
    ylab = 'Model gust error (MOD-OBS) [m/s]'
    # gust error vs obs gust initial
    ax = axes[0,0]
    if i_fancy_colors:
        ax.scatter(obs, err_init, color=get_point_col(obs, err_init), marker=".")
    else:
        ax.scatter(obs, err_init, color='black', marker=".")
    for qi in range(0,len(qs)):
        ax.plot(mp_x, mp_y_init_og[:,qi], color=qscol[qi], linestyle=qslst[qi], linewidth=qslwd[qi])
    ax.axhline(y=0,c='grey')
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_title('original')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.text(xmax-0.3*(xmax-xmin), ymax-0.10*(ymax-ymin), 'rmse '+str(np.round(rmse_init,3)), color='red')
    ax.grid()
    # gust error vs obs gust 
    ax = axes[1,0]
    if i_fancy_colors:
        ax.scatter(obs, err, color=get_point_col(obs, err), marker=".")
    else:
        ax.scatter(obs, err, color='black', marker=".")
    for qi in range(0,len(qs)):
        ax.plot(mp_x, mp_y_og[:,qi], color=qscol[qi], linestyle=qslst[qi], linewidth=qslwd[qi])
    ax.axhline(y=0,c='grey')
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_title('new')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.text(xmax-0.3*(xmax-xmin), ymax-0.10*(ymax-ymin), 'rmse '+str(np.round(rmse,3)), color='red')
    ax.grid()


    xlab = 'Obs mean wind [m/s]'
    ylab = 'Model gust error (MOD-OBS) [m/s]'
    # model mean wind init
    ax = axes[0,1]
    if i_fancy_colors:
        ax.scatter(obs_mean, err_init, color=get_point_col(obs_mean, err_init), marker=".")
    else:
        ax.scatter(obs_mean, err_init, color='black', marker=".")
    for qi in range(0,len(qs)):
        ax.plot(mp_x, mp_y_om_init[:,qi], color=qscol[qi], linestyle=qslst[qi], linewidth=qslwd[qi])
    ax.axhline(y=0,c='grey')
    ax.set_xlim(xmin,xmax/2)
    ax.set_ylim(ymin,ymax)
    ax.set_title('original')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid()
    # model mean wind 
    ax = axes[1,1]
    if i_fancy_colors:
        ax.scatter(obs_mean, err, color=get_point_col(obs_mean, err), marker=".")
    else:
        ax.scatter(obs_mean, err, color='black', marker=".")
    for qi in range(0,len(qs)):
        ax.plot(mp_x, mp_y_om[:,qi], color=qscol[qi], linestyle=qslst[qi], linewidth=qslwd[qi])
    ax.axhline(y=0,c='grey')
    ax.set_xlim(xmin,xmax/2)
    ax.set_ylim(ymin,ymax)
    ax.set_title('new')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid()


    xlab = 'Model gust (MOD) [m/s]'
    ylab = 'Model gust error (MOD-OBS) [m/s]'
    # gust error vs model gust initial
    ax = axes[0,2]
    if i_fancy_colors:
        ax.scatter(gust_init, err_init, color=get_point_col(gust_init, err_init), marker=".")
    else:
        ax.scatter(gust_init, err_init, color='black', marker=".")
    for qi in range(0,len(qs)):
        ax.plot(mp_x, mp_y_mg_init[:,qi], color=qscol[qi], linestyle=qslst[qi], linewidth=qslwd[qi])
    ax.axhline(y=0,c='grey')
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_title('original')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid()
    # gust error vs model gust
    ax = axes[1,2]
    if i_fancy_colors:
        ax.scatter(gust, err, color=get_point_col(gust, err), marker=".")
    else:
        ax.scatter(gust, err, color='black', marker=".")
    for qi in range(0,len(qs)):
        ax.plot(mp_x, mp_y_mg[:,qi], color=qscol[qi], linestyle=qslst[qi], linewidth=qslwd[qi])
    ax.axhline(y=0,c='grey')
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_title('new')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid()



    xlab = 'Model mean wind [m/s]'
    ylab = 'Model gust error (MOD-OBS) [m/s]'
    # model mean wind init
    ax = axes[0,3]
    if i_fancy_colors:
        ax.scatter(model_mean, err_init, color=get_point_col(model_mean, err_init), marker=".")
    else:
        ax.scatter(model_mean, err_init, color='black', marker=".")
    for qi in range(0,len(qs)):
        ax.plot(mp_x, mp_y_mm_init[:,qi], color=qscol[qi], linestyle=qslst[qi], linewidth=qslwd[qi])
    ax.axhline(y=0,c='grey')
    ax.set_xlim(xmin,xmax/2)
    ax.set_ylim(ymin,ymax)
    ax.set_title('original')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid()
    # model mean wind 
    ax = axes[1,3]
    if i_fancy_colors:
        ax.scatter(model_mean, err, color=get_point_col(model_mean, err), marker=".")
    else:
        ax.scatter(model_mean, err, color='black', marker=".")
    for qi in range(0,len(qs)):
        ax.plot(mp_x, mp_y_mm[:,qi], color=qscol[qi], linestyle=qslst[qi], linewidth=qslwd[qi])
    ax.axhline(y=0,c='grey')
    ax.set_xlim(xmin,xmax/2)
    ax.set_ylim(ymin,ymax)
    ax.set_title('new')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid()


    # mean wind error
    xlab = 'Obs mean wind [m/s]'
    ylab = 'Model mean wind error (MOD-OBS) [m/s]'
    ax = axes[0,4]
    ax.scatter(obs_mean, err_mean, color=get_point_col(obs_mean, err_mean), marker=".")
    for qi in range(0,len(qs)):
        ax.plot(mp_x, mp_y_mean[:,qi], color=qscol[qi], linestyle=qslst[qi], linewidth=qslwd[qi])
    ax.axhline(y=0,c='grey')
    ax.set_xlim(xmin,xmax/2)
    ax.set_ylim(ymin,ymax)
    ax.text(xmax/2-0.3*(xmax/2-xmin), ymax-0.10*(ymax-ymin), 'rmse '+str(np.round(rmse_mean,3)), color='red')
    ax.set_title('mean wind error')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid()


    ## obs gust vs model gust initial
    #xlab = 'Model gust (MOD) [m/s]'
    #ylab = 'Obs gust (OBS) [m/s]'
    #ax = axes[1,4]
    #ax.scatter(gust, obs, color=get_point_col(gust, obs), marker=".")
    #for qi in range(0,len(qs)):
    #    ax.plot(mp_x, mp_y_gg[:,qi], color=qscol[qi], linestyle=qslst[qi], linewidth=qslwd[qi])
    ##ax.plot(mp_x, mp_y_gg, color='orange')
    #ax.axhline(y=0,c='grey')
    #ax.set_xlim(0,xmax)
    #ax.set_ylim(0,xmax)
    #ax.set_title('gust vs gust new')
    #ax.set_xlabel(xlab)
    #ax.set_ylabel(ylab)
    #ax.grid()

    # gust error vs mean wind error
    xlab = 'Model mean wind error (MOD-OBS) [m/s]'
    ylab = 'Model gust error (MOD-OBS) [m/s]'
    ax = axes[1,4]
    ax.scatter(err_mean, err, color=get_point_col(err_mean, err, xmin=-60), marker=".")
    for qi in range(0,len(qs)):
        ax.plot(mp_x_me, mp_y_geme[:,qi], color=qscol[qi], linestyle=qslst[qi], linewidth=qslwd[qi])
    ax.axhline(y=0,c='grey')
    ax.set_xlim(ymin,ymax)
    ax.set_ylim(ymin,ymax)
    ax.set_title('gust err vs mean err')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid()

    print('rmse '+str(np.round(rmse,6)))

    #plt.tight_layout()
    plt.subplots_adjust(left=0.04,bottom=0.08,right=0.99,top=0.9,wspace=0.23,hspace=0.3)




import os
import numpy as np
import matplotlib.pyplot as plt
from training_functions import rotate

def draw_grid_wind_plot(CN, data):
    # create directories
    if not os.path.exists(CN.plot_path):
        os.mkdir(CN.plot_path)

    limit = 35

    # process data
    best_model_mean = data['best_model_mean'].flatten()
    best_model_mean = best_model_mean[np.isnan(best_model_mean) == False]
    centre_model_mean = data['centre_model_mean'].flatten()
    centre_model_mean = centre_model_mean[np.isnan(centre_model_mean) == False]
    print(best_model_mean.shape)
    print(centre_model_mean.shape)
    # draw plot
    plt.scatter(centre_model_mean, best_model_mean,
                marker=".",
                color=get_point_col(centre_model_mean, best_model_mean))
    ax = plt.gca()
    ax.set_xlim((0,limit))
    ax.set_ylim((0,limit))
    ax.set_xlabel('centre grid point')
    ax.set_ylabel('best fitting grid point')
    #ax.grid()
    draw_error_grid(limit, limit, ax)
    ax.set_aspect('equal')
    draw_error_percentile_lines(centre_model_mean, best_model_mean, ax, rot_angle=-np.pi/4)
    #plt.show()
    plot_name = CN.plot_path + 'grid_point_wind.png'
    plt.savefig(plot_name)


def get_point_col(vals1, vals2, xmin=0, xmax=100, ymin=-100, ymax=100):
    dx = 1
    dy = 1
    dx = dx*1
    dy = dy*1
    xseq = np.arange(np.floor(xmin),np.ceil(xmax),dx)
    yseq = np.arange(np.floor(ymin),np.ceil(ymax),dy)
    H, xseq, yseq = np.histogram2d(vals1, vals2, bins=(xseq,yseq), normed=True)
    inds1 = np.digitize(vals1, xseq) - 1
    inds2 = np.digitize(vals2, yseq) - 1
    try:
        result = H[inds1,inds2]
    except:
        print('WARNING: Troubles setting colors in get_point_col.' + \
                'Too large values.')
        result = 0.

    cmap = plt.cm.get_cmap('gray')

    result = np.sqrt(np.sqrt(result/np.max(result)))
    col = cmap(result)
    return(col)



def draw_error_percentile_lines(x, y, ax, draw_legend=True, loc=0, rot_angle=0):

    # PREPARE PERCENTILE LINES
    ###############
    #qs = [10,50,90]
    #qscol = ['red','orange','red']
    #qslst = ['--','-','--']
    #qslwd = [1.5,2,1.5]

    #qs = [10,25,50,75,90]
    #qscol = ['red','blue','orange','blue','red']
    #qslst = ['--','--','-','--','--']
    #qslwd = [1.5,1.5,2,1.5,1.5]

    qs = [10,25,50,'mean',75,90]
    qscol = ['red','blue','orange','orange','blue','red']
    qslst = ['--','--','-','--','--','--']
    qslwd = [1.5,1.5,2,2,1.5,1.5]

    dmp = 2. # = dx
    ###############

    xy = np.row_stack( [x, y] )
    xy_rot = rotate(xy, rot_angle)

    speed = xy_rot[0,:]
    error = xy_rot[1,:]

    mp_borders = np.arange(np.floor(0),np.ceil(np.max(speed)),dmp)
    mp_x = mp_borders[:-1] + np.diff(mp_borders)/2
    mgog = np.full((len(mp_x),len(qs)),np.nan)
    for qi in range(0,len(qs)):
        for i in range(0,len(mp_x)):
            # model gust vs obs gust
            inds = (speed > mp_borders[i]) & (speed <= mp_borders[i+1])
            if np.sum(inds) > 20:
                if qs[qi] == 'mean':
                    mgog[i,qi] = np.mean(error[inds])
                else:
                    mgog[i,qi] = np.percentile(error[inds],q=qs[qi])

    lines = []
    for qi in range(0,len(qs)):
        # backtransform percentile lines
        line_rot = np.row_stack( [mp_x, mgog[:,qi]] )
        line = rotate(line_rot, -rot_angle)
        # draw lines
        line, = ax.plot(line[0,:], line[1,:], color=qscol[qi],
                linestyle=qslst[qi], linewidth=qslwd[qi], label=qs[qi])
        lines.append(line)
    if draw_legend:
        ax.legend(handles=lines, loc=loc, title='percentiles')

def draw_error_grid(xmax, ymax, ax):
    ax.plot([0,xmax], [0,ymax], color='k', linewidth=0.5)
    dy = 10/np.sqrt(2)*2
    for i in range(-5,6):
        if i is not 0:
            ax.plot([0,xmax], [i*dy,ymax+i*dy], color='grey', linewidth=0.5)


def draw_1_1_scatter(xvals, yvals, xlims, ylims,
                    xlab, ylab, title, ax, draw_legend=True, legend_loc=0):
    ax.scatter(xvals, yvals, color=get_point_col(xvals, yvals), marker=".")
    draw_error_grid(xlims[1], ylims[1], ax)
    draw_error_percentile_lines(xvals, yvals, ax, draw_legend, legend_loc,
                                rot_angle=-np.pi/4)
    ax.axhline(y=0,c='grey')
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    # calcualte some scores
    #categ_thrshs = [20,40]
    categ_thrshs = [int(xlims[1]*1/3),int(xlims[1]*2/3)]
    xy = np.row_stack( [xvals, yvals] )
    xy_rot = rotate(xy, -np.pi/4)
    error = xy_rot[1,:]
    rmse = np.sqrt(np.mean(error**2))
    corr = np.corrcoef(xvals, yvals)[1,0]
    me   = np.mean(error) # positive values means model overestimates
    #mae  = np.mean(np.abs(error))
    pods = np.zeros(len(categ_thrshs))
    fars = np.zeros(len(categ_thrshs))
    for tI,thrsh in enumerate(categ_thrshs):
        pods[tI] = np.sum(yvals[xvals > thrsh] >= thrsh)/np.sum(xvals > thrsh)
        fars[tI] = np.sum(xvals[yvals > thrsh] <  thrsh)/np.sum(yvals > thrsh)
    errors = {'rmse':rmse,'corr':corr,'me':me,
                'pod'+str(categ_thrshs[0]):pods[0],
                'pod'+str(categ_thrshs[1]):pods[1],
                'far'+str(categ_thrshs[0]):fars[0],
                'far'+str(categ_thrshs[1]):fars[1]}
    # print scores on plot
    col = 'k'
    offs = 0.05
    ax.text(0.02*(xlims[1]-xlims[0]),
            ylims[1]-1*offs*(ylims[1]-ylims[0]),
            'pod({}) {}'.format(str(categ_thrshs),
            str(np.round(pods,2))), color=col)
    ax.text(0.02*(xlims[1]-xlims[0]),
            ylims[1]-2*offs*(ylims[1]-ylims[0]),
            'far  ({}) {}'.format(str(categ_thrshs),
            str(np.round(fars,2))), color=col)
    ax.text(0.02*(xlims[1]-xlims[0]),
            ylims[1]-3*offs*(ylims[1]-ylims[0]),
            'rmse '+str(np.round(rmse,3)), color=col)
    ax.text(0.02*(xlims[1]-xlims[0]),
            ylims[1]-4*offs*(ylims[1]-ylims[0]),
            'corr   '+str(np.round(corr,3)), color=col)
    ax.text(0.02*(xlims[1]-xlims[0]),
            ylims[1]-5*offs*(ylims[1]-ylims[0]),
            'me    '+str(np.round(me,3)), color=col)
    #ax.text(0.02*(xlims[1]-xlims[0]),
    #        ylims[1]-6*offs*(ylims[1]-ylims[0]),
    #        'mae  '+str(np.round(mae,3)), color=col)
    return(errors)


def draw_error_scatter(mod, obs, xlims, ylims,
                    xlab, ylab, title, ax, draw_legend=True, legend_loc=0):
    error = mod - obs
    ax.scatter(obs, error, color=get_point_col(obs, error), marker=".")
    ax.grid()
    draw_error_percentile_lines(obs, error, ax, draw_legend, legend_loc)
    ax.axhline(y=0,c='grey')
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)


def plot_type1(obs, gust, gust_ref, obs_mean, mod_mean):

    xlims = (0,60)
    ylims = (0,60)
    xlims_mean = (0,30)
    ylims_mean = (0,30)
    ylims_err = (-30,30)
    ylims_err_mean = (-15,15)

    fig,axes = plt.subplots(2,3, figsize=(13.0,9.2))
    plt.subplots_adjust(left=0.05,bottom=0.08,right=0.99,top=0.9,
                        wspace=0.23,hspace=0.3)

    ##########################################################################
    xlab = 'OBS gust [m/s]'
    ylab = 'MOD gust [m/s]'
    title = 'reference MOD gust vs OBS gust'
    ax = axes[0,0]
    errors_ref = draw_1_1_scatter(obs, gust_ref, xlims, ylims,
                    xlab, ylab, title, ax, legend_loc=4)

    ##########################################################################
    xlab = 'OBS gust [m/s]'
    ylab = 'MOD gust [m/s]'
    title = 'MOD gust vs OBS gust'
    ax = axes[0,1]
    errors = draw_1_1_scatter(obs, gust, xlims, ylims,
                    xlab, ylab, title, ax, draw_legend=False)

    ##########################################################################
    xlab = 'OBS mean wind [m/s]'
    ylab = 'MOD mean wind [m/s]'
    title = 'MOD wind vs OBS wind'
    ax = axes[0,2]
    draw_1_1_scatter(obs_mean, mod_mean, xlims_mean, ylims_mean,
                    xlab, ylab, title, ax, draw_legend=False)

    ##########################################################################
    xlab = 'OBS gust [m/s]'
    ylab = 'reference MOD gust error [m/s]'
    title = 'reference MOD gust error vs OBS gust'
    ax = axes[1,0]
    draw_error_scatter(gust_ref, obs, xlims, ylims_err,
                    xlab, ylab, title, ax, draw_legend=False)

    ##########################################################################
    xlab = 'OBS gust [m/s]'
    ylab = 'MOD gust error [m/s]'
    title = 'MOD gust error vs OBS gust'
    ax = axes[1,1]
    draw_error_scatter(gust, obs, xlims, ylims_err,
                    xlab, ylab, title, ax, draw_legend=False)

    ##########################################################################
    xlab = 'OBS mean wind [m/s]'
    ylab = 'MOD mean wind error [m/s]'
    title = 'MOD mean wind error vs OBS mean wind'
    ax = axes[1,2]
    draw_error_scatter(mod_mean, obs_mean, xlims_mean, ylims_err_mean,
                    xlab, ylab, title, ax, draw_legend=False)


    return((errors_ref, errors))



























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




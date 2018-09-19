import numpy as np
import pandas as pd
import globals as G
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import timedelta
from scipy.spatial import KDTree
from netCDF4 import Dataset
import time

def load_var_at_station_from_nc(nc_path, var_name, sel_stat):
    # GET STATION i AND j AND fort_stat INDEX
    station_file = "/users/heimc/stations/all_stations.lst"
    station_list = np.genfromtxt(station_file, skip_header=1, dtype=np.str)
    headers = station_list[0,:]
    station_list = station_list[1:,:]
    #print(headers)
    sel_stat_ind = station_list[:,0] == sel_stat
    print(station_list[sel_stat_ind,:])
    # final indices
    i_ind = station_list[sel_stat_ind,8].astype(np.int) - 1
    j_ind = station_list[sel_stat_ind,9].astype(np.int) - 1
    print('select values at grind points: i ' + str(i_ind) + ' j ' + str(j_ind))

    ## LOAD NC FILE DATA
    #print('##############')

    ncf = Dataset(nc_path, 'r')
    ndim = np.ndim(ncf[var_name][:])
    if ndim == 3:
        var_nc = ncf[var_name][:,j_ind,i_ind].flatten()
    elif ndim == 4:
        kind = 79
        var_nc = ncf[var_name][:,kind,j_ind,i_ind].flatten()

    return(var_nc)


def check_prerequisites(data, prerequisites, hist_tag):
    for prereq in prerequisites:
        if prereq not in data[G.HIST]:
            raise ValueError('Functions ' + prereq + \
            ' needs to be applied to data to run ' + hist_tag)
    data[G.HIST].append(hist_tag)
    return(data)


def custom_resampler(array_like):
    return(array_like[-1])

def calc_model_fields(data, i_model_fields):
    
    offset = timedelta(hours=1)

    """
    Calculate model fields (also gusts according to all methods)
    and aggregates model values to hourly steps.

    INPUT
    data:           dictionary containing data
    i_model_fields: list containing string of model fields to calculate:
                    for options see globals.py section 'Model Fields':

    OUTPUT
    data:           added FIELDS entry to data (data[G.MODEL][...][G.FIELDS]) according to i_model_fields
    """
    hist_tag = 'calc_model_fields'
    prerequisites = ['01_prep_obs', '02_prep_model']
    data = check_prerequisites(data, prerequisites, hist_tag)

    # loop through all stations
    for stat in data[G.STAT_NAMES]:

        # add fields dictionary entry
        data[G.MODEL][G.STAT][stat][G.FIELDS] = {}

        for lm_run in list(data[G.MODEL][G.STAT][stat][G.RAW].keys()):
            
            curr_raw = data[G.MODEL][G.STAT][stat][G.RAW][lm_run]
    
            all_fields = []
            for field_name in i_model_fields:

                if field_name in G.FIELDS_GUST:
                    # Calculate gusts for various methods

                    if field_name == G.GUST_MIX_COEF_LINEAR:
            
                        tcm = curr_raw['tcm']
                        zcm = tcm
                        zcm[zcm < 5E-4] = 5E-4
                        zsqcm = np.sqrt(zcm)
                        zvp10 = curr_raw['zvp10']
                        gust = zvp10 + 3.0 * 2.4 * zsqcm * zvp10
                        #gust = zvp10 + 3.615 * zsqcm * zvp10 # Pompa tuning
                        gust = gust.to_frame()
            
                    elif field_name == G.GUST_MIX_COEF_NONLIN:
            
                        tcm = curr_raw['tcm']
                        zcm = tcm
                        zcm[zcm < 5E-4] = 5E-4
                        zsqcm = np.sqrt(zcm)
                        zvp10 = curr_raw['zvp10']
                        gust = zvp10 + (3.0 * 2.4 + 0.09 * zvp10) * zsqcm * zvp10
                        gust = gust.to_frame()

                    elif field_name == G.GUST_ICON:
                        
                        ugn = 7.71
                        hpbl = 1000
                        g = 9.80665
                        Rd = 287.05
                        etv = 0.6078
                        cp = 1005.0
                        kappa = Rd/cp

                        umlev = curr_raw['ul1'].values
                        vmlev = curr_raw['vl1'].values
                        ps = curr_raw['ps'].values
                        qvflx = curr_raw['qvflx'].values
                        shflx = curr_raw['shflx'].values
                        z0 = curr_raw['z0'].values
                        Tskin = curr_raw['Tskin'].values
                        Tmlev = curr_raw['Tl1'].values
                        qvmlev = curr_raw['qvl1'].values
                        phimlev = curr_raw['phil1'].values

                        # density
                        rho = ps / ( Rd*Tmlev * ( 1 + etv*qvmlev ) )

                        # buoyancy
                        buoy = g * ( - etv*qvflx - shflx/( Tskin*cp ) ) / rho

                        # surface stress
                        zcdn = ( kappa / np.log( 1 + phimlev/(g*z0) ) )**2
                        dua = np.sqrt( np.maximum( 0.1**2, umlev**2 + vmlev**2 ) )
                        ustr = rho*umlev*dua*zcdn
                        vstr = rho*vmlev*dua*zcdn

                        # friction velocity
                        ustar2 = np.sqrt( ustr**2 + vstr**2 ) / rho 
                        wstar2 = ( buoy[buoy > 0]*hpbl )**(2/3)
                        ustar2[buoy > 0] = ustar2[buoy > 0] + 2E-3*wstar2
                        ustar = np.maximum( np.sqrt(ustar2), 0.0001 )

                        # wind gust
                        zvp10 = curr_raw['zvp10']
                        idl = -hpbl*kappa*buoy/ustar**3
                        gust = zvp10
                        greater0 = idl >= 0
                        gust[greater0] = gust[greater0] + ustar[greater0]*ugn
                        smaller0 = idl < 0
                        gust[smaller0] = gust[smaller0] + ustar[smaller0]*ugn * (1 - 0.5/12*idl[smaller0])**(1/3)
                        #gust = zvp10 + ustar*ugn*( 1 + 0.5/12*hpbl*kappa*buoy/ustar**3 )**(1/3)
                        gust = gust.to_frame()
                        
                    elif field_name in G.FIELDS_BRA_GUST:
                        if field_name == G.GUST_BRASSEUR_ESTIM:
                            gust_field_name = 'zv_bra_es'
                        elif field_name == G.GUST_BRASSEUR_LOBOU:
                            gust_field_name = 'zv_bra_lb'
                        elif field_name == G.GUST_BRASSEUR_UPBOU:
                            gust_field_name = 'zv_bra_ub'
            
                        gust = curr_raw[gust_field_name].to_frame()

                    # resample to maximum hourly value
                    # time index shifted backward 10 seconds make sure resampling has expected behavior
                    gust = gust.shift(freq='-10s')
                    field = gust.resample('H', loffset=offset).max()
                    #field = gust.resample('H', how=lambda x: np.percentile(x, q=80))

                elif field_name in G.FIELDS_BRA_KVAL:

                    if field_name == G.KVAL_BRASSEUR_ESTIM:
                        gust_field_name = 'zv_bra_es'
                        k_field_name = 'k_bra_es'
                    elif field_name == G.KVAL_BRASSEUR_LOBOU:
                        gust_field_name = 'zv_bra_lb'
                        k_field_name = 'k_bra_lb'
                    elif field_name == G.KVAL_BRASSEUR_UPBOU:
                        gust_field_name = 'zv_bra_ub'
                        k_field_name = 'k_bra_ub'

                    gust = curr_raw[gust_field_name]
                    # time index shifted backward 10 seconds make sure resampling has expected behavior
                    gust = gust.shift(freq='-10s')
                    maxid = gust.resample('H', convention='end').agg({gust_field_name: np.argmax})
                    values = curr_raw[k_field_name][maxid].values
                    raise NotImplementedError()
                    field = pd.DataFrame(values, index=maxid.index.levels[1], columns=[field_name]).astype(int)

                elif field_name == G.MODEL_MEAN_WIND:
                    
                    field = curr_raw['zvp10'].to_frame()
                    # time index shifted backward 10 seconds make sure resampling has expected behavior
                    field = field.shift(freq='-10s')
                    field = field.resample('H', loffset=offset).mean()

                elif field_name == G.MEAN_WIND_INST:
                    
                    field = curr_raw['zvp10'].to_frame()
                    # time index shifted backward 10 seconds make sure resampling has expected behavior
                    field = field.shift(freq='-10s')
                    field = field.resample('H', loffset=offset).apply(custom_resampler)

                else:
                    raise ValueError('Unknown user input "' + field_name + '" in list i_gust_fields!')
        
                field.columns = [field_name]
                all_fields.append(field)
            
            all_fields = pd.concat(all_fields, axis=1)

            # save in data
            data[G.MODEL][G.STAT][stat][G.FIELDS][lm_run] = all_fields

    return(data)
    

def join_model_and_obs(data):

    """
    Combines the hourly calculated model fields with the hourly observation.
    Puts this into data-sub-dictionary G.BOTH

    INPUT
    data:           dictionary containing the dict data[G.MODEL][...][G.FIELDS]

    OUTPUT
    data:           added G.BOTH entry to data dictionary
    """
    hist_tag = 'join_model_and_obs'
    prerequisites = ['01_prep_obs', '02_prep_model',
                    'calc_model_fields']
    data = check_prerequisites(data, prerequisites, hist_tag)

    data[G.BOTH] = {G.STAT:{}}

    # loop through all stations
    for stat in data[G.STAT_NAMES]:

        data[G.BOTH][G.STAT][stat] = {}

        for lm_run in list(data[G.MODEL][G.STAT][stat][G.FIELDS].keys()):
            
            model = data[G.MODEL][G.STAT][stat][G.FIELDS][lm_run]
            obs = data[G.OBS][G.STAT][stat]

            both = pd.concat([model, obs], axis=1, join='inner')
            data[G.BOTH][G.STAT][stat][lm_run] = both
            
    return(data)


def join_model_runs(data):

    """
    Assumes that data already contains data[G.BOTH] !
    Joins the dataframes of all model runs (with different lead time) into one dataframe.

    INPUT
    data:           dictionary containing the dict data[G.BOTH]

    OUTPUT
    data:           combined all data[G.BOTH][...][model run].df into data[G.BOTH][...].df
    """
    hist_tag = 'join_model_runs'
    prerequisites = ['01_prep_obs', '02_prep_model',
                    'calc_model_fields', 'join_model_and_obs']
    data = check_prerequisites(data, prerequisites, hist_tag)

    # loop through all stations
    for stat in data[G.STAT_NAMES]:

        dfs = []
        for lm_run in list(data[G.MODEL][G.STAT][stat][G.FIELDS].keys()):
            
            dfs.append(data[G.BOTH][G.STAT][stat][lm_run])

        df = pd.concat(dfs, axis=0)
        data[G.BOTH][G.STAT][stat] = df
            
    return(data)


def join_all_stations(data):

    """
    Assumes that data already contains data[G.BOTH] and model runs with different lead time joined !
    Joins the dataframes of all stations

    INPUT
    data:           dictionary containing the dict data[G.BOTH]

    OUTPUT
    data:           combined all data[G.BOTH][G.STAT][...].df to data[G.BOTH][G.ALL_STAT].df
    """
    hist_tag = 'join_all_stations'
    prerequisites = ['01_prep_obs', '02_prep_model',
                    'calc_model_fields', 'join_model_and_obs', 'join_model_runs']
    data = check_prerequisites(data, prerequisites, hist_tag)

    dfs = []
    # loop through all stations
    for stat in data[G.STAT_NAMES]:
        dfs.append(data[G.BOTH][G.STAT][stat])

    df = pd.concat(dfs, axis=0)
    data[G.BOTH][G.ALL_STAT] = df
            
    return(data)



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
    dx = 0.5
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




import numpy as np
import pandas as pd
import globals as G
from sklearn import metrics
import matplotlib.pyplot as plt


def check_prerequisites(data, prerequisites, hist_tag):
    for prereq in prerequisites:
        if prereq not in data[G.HIST]:
            raise ValueError('Functions ' + prereq + \
            ' needs to be applied to data to run ' + hist_tag)
    data[G.HIST].append(hist_tag)
    return(data)



def calc_model_fields(data, i_model_fields):

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
                    field = gust.resample('H').max()
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
                    maxid = gust.resample('H').agg({gust_field_name: np.argmax})
                    #print(maxid)
                    #quit()
                    values = curr_raw[k_field_name][maxid].values
                    field = pd.DataFrame(values, index=maxid.index.levels[1], columns=[field_name]).astype(int)

                elif field_name == G.MODEL_MEAN_WIND:
                    
                    field = curr_raw['zvp10'].to_frame()
                    field = field.resample('H').mean()

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





def plot_error(obs, model_mean, obs_mean, gust, gust_init):
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
    

    err = gust - obs
    err_init = gust_init - obs
    err_mean = model_mean - obs_mean
    rmse = np.sqrt(np.sum(err**2)/len(err))
    rmse_init = np.sqrt(np.sum(err_init**2)/len(err))

    # calculate median
    dmp = 1
    mp_borders = np.arange(np.floor(xmin),np.ceil(xmax),dmp)
    mp_x = mp_borders[:-1] + np.diff(mp_borders)/2
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
            inds = (model_mean > mp_borders[i]) & (model_mean <= mp_borders[i+1])
            inds = (obs_mean > mp_borders[i]) & (obs_mean <= mp_borders[i+1])
            if np.sum(inds) > 10:
                mp_y_mean[i,qi] = np.nanpercentile(err_mean[inds],q=qs[qi])
            # obs gust vs model gust
            inds = (gust > mp_borders[i]) & (gust <= mp_borders[i+1])
            if np.sum(inds) > 10:
                mp_y_gg[i,qi] = np.percentile(obs[inds],q=qs[qi])

    xlab = 'Observed gust (OBS) [m/s]'
    ylab = 'Model gust error (MOD-OBS) [m/s]'
    # gust error vs obs gust initial
    ax = axes[0,0]
    ax.scatter(obs, err_init, color='black', marker=".")
    for qi in range(0,len(qs)):
        ax.plot(mp_x, mp_y_init_og[:,qi], color=qscol[qi], linestyle=qslst[qi], linewidth=qslwd[qi])
    ax.axhline(y=0,c='grey')
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_title('original')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.text(xmax-0.3*(xmax-xmin), ymax-0.10*(ymax-ymin), 'rmse '+str(np.round(rmse_init,2)), color='red')
    ax.grid()
    # gust error vs obs gust 
    ax = axes[1,0]
    ax.scatter(obs, err, color='black', marker=".")
    for qi in range(0,len(qs)):
        ax.plot(mp_x, mp_y_og[:,qi], color=qscol[qi], linestyle=qslst[qi], linewidth=qslwd[qi])
    ax.axhline(y=0,c='grey')
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_title('new')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.text(xmax-0.3*(xmax-xmin), ymax-0.10*(ymax-ymin), 'rmse '+str(np.round(rmse,2)), color='red')
    ax.grid()


    xlab = 'Obs mean wind [m/s]'
    ylab = 'Model gust error (MOD-OBS) [m/s]'
    # model mean wind init
    ax = axes[0,1]
    ax.scatter(obs_mean, err_init, color='black', marker=".")
    for qi in range(0,len(qs)):
        ax.plot(mp_x, mp_y_om_init[:,qi], color=qscol[qi], linestyle=qslst[qi], linewidth=qslwd[qi])
    #ax.plot(mp_x, mp_y_om_init, color='orange')
    ax.axhline(y=0,c='grey')
    ax.set_xlim(xmin,xmax/2)
    ax.set_ylim(ymin,ymax)
    ax.set_title('original')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid()
    # model mean wind 
    ax = axes[1,1]
    ax.scatter(obs_mean, err, color='black', marker=".")
    for qi in range(0,len(qs)):
        ax.plot(mp_x, mp_y_om[:,qi], color=qscol[qi], linestyle=qslst[qi], linewidth=qslwd[qi])
    #ax.plot(mp_x, mp_y_om, color='orange')
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
    ax.scatter(gust_init, err_init, color='black', marker=".")
    for qi in range(0,len(qs)):
        ax.plot(mp_x, mp_y_mg_init[:,qi], color=qscol[qi], linestyle=qslst[qi], linewidth=qslwd[qi])
    #ax.plot(mp_x, mp_y_mg_init, color='orange')
    ax.axhline(y=0,c='grey')
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_title('original')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid()
    # gust error vs model gust
    ax = axes[1,2]
    ax.scatter(gust, err, color='black', marker=".")
    for qi in range(0,len(qs)):
        ax.plot(mp_x, mp_y_mg[:,qi], color=qscol[qi], linestyle=qslst[qi], linewidth=qslwd[qi])
    #ax.plot(mp_x, mp_y_mg, color='orange')
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
    ax.scatter(model_mean, err_init, color='black', marker=".")
    for qi in range(0,len(qs)):
        ax.plot(mp_x, mp_y_mm_init[:,qi], color=qscol[qi], linestyle=qslst[qi], linewidth=qslwd[qi])
    #ax.plot(mp_x, mp_y_mm_init, color='orange')
    ax.axhline(y=0,c='grey')
    ax.set_xlim(xmin,xmax/2)
    ax.set_ylim(ymin,ymax)
    ax.set_title('original')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid()
    # model mean wind 
    ax = axes[1,3]
    ax.scatter(model_mean, err, color='black', marker=".")
    for qi in range(0,len(qs)):
        ax.plot(mp_x, mp_y_mm[:,qi], color=qscol[qi], linestyle=qslst[qi], linewidth=qslwd[qi])
    #ax.plot(mp_x, mp_y_mm, color='orange')
    ax.axhline(y=0,c='grey')
    ax.set_xlim(xmin,xmax/2)
    ax.set_ylim(ymin,ymax)
    ax.set_title('new')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid()




    # mean wind error
    xlab = 'Model mean wind [m/s]'
    ylab = 'Model mean wind error (MOD-OBS) [m/s]'
    ax = axes[0,4]
    #ax.scatter(model_mean, err_mean, color='black', marker=".")
    ax.scatter(obs_mean, err_mean, color='black', marker=".")
    for qi in range(0,len(qs)):
        ax.plot(mp_x, mp_y_mean[:,qi], color=qscol[qi], linestyle=qslst[qi], linewidth=qslwd[qi])
    #ax.plot(mp_x, mp_y_mean, color='orange')
    ax.axhline(y=0,c='grey')
    ax.set_xlim(xmin,xmax/2)
    ax.set_ylim(ymin,ymax)
    ax.set_title('mean wind error')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid()


    # obs gust vs model gust initial
    xlab = 'Model gust (MOD) [m/s]'
    ylab = 'Obs gust (OBS) [m/s]'
    ax = axes[1,4]
    ax.scatter(gust, obs, color='black', marker=".")
    for qi in range(0,len(qs)):
        ax.plot(mp_x, mp_y_gg[:,qi], color=qscol[qi], linestyle=qslst[qi], linewidth=qslwd[qi])
    #ax.plot(mp_x, mp_y_gg, color='orange')
    ax.axhline(y=0,c='grey')
    ax.set_xlim(0,xmax)
    ax.set_ylim(0,xmax)
    ax.set_title('gust vs gust new')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid()

    print('rmse '+str(np.round(rmse,2)))

    #plt.tight_layout()
    plt.subplots_adjust(left=0.04,bottom=0.08,right=0.99,top=0.9,wspace=0.23,hspace=0.3)



def scale(var):
    mean_var = np.mean(var)
    var = var - mean_var
    sd_var = np.std(var)
    var = var/sd_var
    return(var)






#def calc_scores(data, i_scores):
#
#    """
#    Calculate scores for the hourly gusts in data
#
#    INPUT
#    data:           dictionary containing [G.MODEL][G.FIELDS] (with calcualted gusts)
#    i_scores:       list containing string of scores to calculate.
#                    for options see globals.py section 'scores':
#
#    OUTPUT
#    data:           added SCORE entry to data according to i_scores (data[SCORE][...][
#    """
#    data[G.SCORE] = {G.STAT:{}}
#
#    data = join_model_runs(data)
#
#    # loop through all stations
#    for stat in data[G.STAT_NAMES]:
#
#        # add score dictionary entry
#        data[G.SCORE][G.STAT][stat] = {}
#
#        for gust_method in G.FIELDS_GUST:
#            if gust_method in data[G.BOTH][G.STAT][stat]:
#                print(gust_method)
#        
#        quit()
#    
#        gust_methods = list(data[G.MODEL][G.STAT][stat][G.FIELDS].keys())
#        #gust_methods = [gm for gm in gust_methods_inp if gm in G.FIELDS_GUST]
#        #if len(gust_methods_inp) != len(gust_methods):
#        #    raise ValueError('Non-gust field in score calculation!')
#
#        # vector of observed gusts
#        gust_obs = data[G.OBS][G.STAT][stat][G.PAR]['VMAX_10M1'].values
#        # vector of observed mean winds
#        wind_obs = data[G.OBS][G.STAT][stat][G.PAR]['FF_10M'].values
#
#        for gust_method in gust_methods:
#
#            # add entry for gust method in score dictionary
#            data[G.MODEL][G.STAT][stat][G.SCORE][gust_method] = {}
#
#            # vector of simulated gusts
#            gust_mod = data[G.MODEL][G.STAT][stat][G.FIELDS][gust_method]
#
#            if gust_method in G.FIELDS_GUST:
#                eval_field = gust_obs
#            else:
#                eval_field = wind_obs
#    
#            for score_name in i_scores:
#                
#                # Vector scores
#                if score_name == G.SCORE_ME:
#                    score = gust_mod - eval_field
#                elif score_name == G.SCORE_AE:
#                    score = np.abs(gust_mod - eval_field)
#
#                # scalar scores
#                elif score_name == G.SCORE_MAE:
#                    score = metrics.mean_absolute_error(eval_field, gust_mod)
#
#                # add score to gust method dictionary
#                data[G.MODEL][G.STAT][stat][G.SCORE][gust_method][score_name] = score
#        
#
#    return(data)








#def remove_obs_nan_in_hourly_fields(data, obs_field):
#
#    """
#    Remove nan in hourly fields:
#    FIELDS
#    OBS
#    according to missing values given in the OBS field 'obs_field'
#
#    INPUT
#    data:           dictionary containing data (with calcualted gusts)
#    obs_field:      string of OBS field name according to which NaN's should be removed
#
#    OUTPUT
#    data:           processed data dictionary
#    """
#    for stat in data[G.STAT_NAMES]:
#        # vector of observed gusts
#        gust_obs = data[G.OBS][G.STAT][stat][G.PAR][obs_field].values
#        mask = np.isnan(gust_obs)
#        keep = ~mask
#
#        # remove in obs
#        obs_fields = data[G.OBS][G.PAR_NAMES]
#        for field in obs_fields:
#            data[G.OBS][G.STAT][stat][G.PAR][field] = data[G.OBS][G.STAT][stat][G.PAR][field][keep]
#
#        # remove in model
#        mod_gust_fields = list(data[G.MODEL][G.STAT][stat][G.FIELDS].keys())
#        for field in mod_gust_fields:
#            data[G.MODEL][G.STAT][stat][G.GUST][field] = data[G.MODEL][G.STAT][stat][G.GUST][field][keep]
#
#    return(data)

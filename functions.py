import numpy as np
import globals as G
from sklearn import metrics


def calc_gusts(data, i_gust_fields):

    """
    Calculate gusts according to various methods

    INPUT
    data:           dictionary containing data
    i_gust_fields:  list containing string of gust method names:
                    for options see globals.py section 'Gust methods':

    OUTPUT
    data:           added GUSTS entry to MODEL according to i_gust_fields
    """

    # number of stations and number of hours in data set
    nstat = len(data[G.STAT_NAMES])
    nhrs = len(data[G.OBS][G.DTS])

    # Prepare index mask to map time step model output to hourly values
    hr_inds = np.zeros((nhrs,360))
    for i in range(0,nhrs):
        hr_inds[i,:] = i*360 + np.arange(0,360)
    hr_inds = hr_inds.astype(np.int)
   
    # loop through all stations
    for stat in data[G.STAT_NAMES]:
        #print(stat)

        # add gust dictionary entry
        data[G.MODEL][G.STAT][stat][G.GUST] = {}
    
        for method in i_gust_fields:
            #print(method)
    
            # Calculate gusts for various methods

            if method == G.GUST_MIX_COEF_LINEAR:
    
                tcm = data[G.MODEL][G.STAT][stat][G.PAR]['tcm']
                zcm = tcm
                zcm[zcm < 5E-4] = 5E-4
                zsqcm = np.sqrt(zcm)
                zvp10 = data[G.MODEL][G.STAT][stat][G.PAR]['zvp10']
                gust = zvp10 + 3.0 * 2.4 * zsqcm * zvp10
    
            elif method == G.GUST_MIX_COEF_NONLIN:
    
                tcm = data[G.MODEL][G.STAT][stat][G.PAR]['tcm']
                zcm = tcm
                zcm[zcm < 5E-4] = 5E-4
                zsqcm = np.sqrt(zcm)
                zvp10 = data[G.MODEL][G.STAT][stat][G.PAR]['zvp10']
                gust = zvp10 + (3.0 * 2.4 + 0.09 * zvp10) * zsqcm * zvp10
                
            elif method == G.GUST_BRASSEUR_ESTIM:
    
                gust = data[G.MODEL][G.STAT][stat][G.PAR]['zv_bra_es']

            elif method == G.GUST_BRASSEUR_LOBOU:
    
                gust = data[G.MODEL][G.STAT][stat][G.PAR]['zv_bra_lb']

            elif method == G.GUST_BRASSEUR_UPBOU:
    
                gust = data[G.MODEL][G.STAT][stat][G.PAR]['zv_bra_ub']

            elif method == G.KVAL_BRASSEUR_ESTIM:

                gust = data[G.MODEL][G.STAT][stat][G.PAR]['zv_bra_es']
                kval = data[G.MODEL][G.STAT][stat][G.PAR]['k_bra_es']

            elif method == G.KVAL_BRASSEUR_LOBOU:

                gust = data[G.MODEL][G.STAT][stat][G.PAR]['zv_bra_lb']
                kval = data[G.MODEL][G.STAT][stat][G.PAR]['k_bra_lb']

            elif method == G.KVAL_BRASSEUR_UPBOU:

                gust = data[G.MODEL][G.STAT][stat][G.PAR]['zv_bra_ub']
                kval = data[G.MODEL][G.STAT][stat][G.PAR]['k_bra_ub']

            else:
                raise ValueError('Unknown user input "' + method + '" in list i_gust_fields!')
    
        
            # Aggregate to hourly values
            hr_max_gusts = np.full((nhrs),np.nan)

            if method in [G.KVAL_BRASSEUR_ESTIM,G.KVAL_BRASSEUR_LOBOU,G.KVAL_BRASSEUR_UPBOU]:

                # find index of maximum brasseur gust and store k value
                for hr in range(0,nhrs):
                    inds = hr_inds[hr]
                    # find maximum gust ind
                    hr_max_gust_ind = np.argmax(gust[inds])
                    hr_max_gusts[hr] = kval[inds][hr_max_gust_ind]
        
            else: # any other field

                # calc and save model max gust
                for hr in range(0,nhrs):
                    inds = hr_inds[hr]
                    # find maximum gust
                    hr_max_gust = np.max(gust[inds])
                    hr_max_gusts[hr] = hr_max_gust
        
            # save in data
            data[G.MODEL][G.STAT][stat][G.GUST][method] = hr_max_gusts

    return(data)
    

#def remove_obs_nan_in_hourly_fields(data, obs_field):
#
#    """
#    Remove nan in hourly fields:
#    GUST
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
#        mod_gust_fields = list(data[G.MODEL][G.STAT][stat][G.GUST].keys())
#        for field in mod_gust_fields:
#            data[G.MODEL][G.STAT][stat][G.GUST][field] = data[G.MODEL][G.STAT][stat][G.GUST][field][keep]
#
#    return(data)



def calc_scores(data, i_scores):

    """
    Calculate scores for the hourly gusts in data

    INPUT
    data:           dictionary containing data (with calcualted gusts)
    i_scores:       list containing string of scores to calculate.
                    for options see globals.py section 'scores':

    OUTPUT
    data:           added SCORE entry to MODEL according to i_scores
    """
    # loop through all stations
    for stat in data[G.STAT_NAMES]:

        # add score dictionary entry
        data[G.MODEL][G.STAT][stat][G.SCORE] = {}
    
        gust_methods = list(data[G.MODEL][G.STAT][stat][G.GUST].keys())

        # vector of observed gusts
        gust_obs = data[G.OBS][G.STAT][stat][G.PAR]['VMAX_10M1'].values

        for gust_method in gust_methods:

            # add entry for gust method in score dictionary
            data[G.MODEL][G.STAT][stat][G.SCORE][gust_method] = {}

            # vector of simulated gusts
            gust_mod = data[G.MODEL][G.STAT][stat][G.GUST][gust_method]
    
            for score_name in i_scores:
                
                # Vector scores
                if score_name == G.SCORE_ME:
                    score = gust_mod - gust_obs
                elif score_name == G.SCORE_AE:
                    score = np.abs(gust_mod - gust_obs)

                # scalar scores
                elif score_name == G.SCORE_MAE:
                    score = metrics.mean_absolute_error(gust_obs, gust_mod)

                # add score to gust method dictionary
                data[G.MODEL][G.STAT][stat][G.SCORE][gust_method][score_name] = score
        

    return(data)

import numpy as np
import globals as G
from sklearn import metrics


def calc_model_fields(data, i_model_fields):

    """
    Calculate model fields (also gusts according to all methods)
    and aggregates model values to hourly steps.

    INPUT
    data:           dictionary containing data
    i_model_fields: list containing string of model fields to calculate:
                    for options see globals.py section 'Model Fields':

    OUTPUT
    data:           added FIELDS entry to MODEL according to i_model_fields
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

        # add fields dictionary entry
        data[G.MODEL][G.STAT][stat][G.FIELDS] = {}
    
        for field_name in i_model_fields:
            #print(field_name)
    
            # Calculate gusts for various methods

            if field_name == G.GUST_MIX_COEF_LINEAR:
    
                tcm = data[G.MODEL][G.STAT][stat][G.PAR]['tcm']
                zcm = tcm
                zcm[zcm < 5E-4] = 5E-4
                zsqcm = np.sqrt(zcm)
                zvp10 = data[G.MODEL][G.STAT][stat][G.PAR]['zvp10']
                gust = zvp10 + 3.0 * 2.4 * zsqcm * zvp10
    
            elif field_name == G.GUST_MIX_COEF_NONLIN:
    
                tcm = data[G.MODEL][G.STAT][stat][G.PAR]['tcm']
                zcm = tcm
                zcm[zcm < 5E-4] = 5E-4
                zsqcm = np.sqrt(zcm)
                zvp10 = data[G.MODEL][G.STAT][stat][G.PAR]['zvp10']
                gust = zvp10 + (3.0 * 2.4 + 0.09 * zvp10) * zsqcm * zvp10
                
            elif field_name == G.GUST_BRASSEUR_ESTIM:
    
                gust = data[G.MODEL][G.STAT][stat][G.PAR]['zv_bra_es']

            elif field_name == G.GUST_BRASSEUR_LOBOU:
    
                gust = data[G.MODEL][G.STAT][stat][G.PAR]['zv_bra_lb']

            elif field_name == G.GUST_BRASSEUR_UPBOU:
    
                gust = data[G.MODEL][G.STAT][stat][G.PAR]['zv_bra_ub']

            elif field_name == G.KVAL_BRASSEUR_ESTIM:

                gust = data[G.MODEL][G.STAT][stat][G.PAR]['zv_bra_es']
                kval = data[G.MODEL][G.STAT][stat][G.PAR]['k_bra_es']

            elif field_name == G.KVAL_BRASSEUR_LOBOU:

                gust = data[G.MODEL][G.STAT][stat][G.PAR]['zv_bra_lb']
                kval = data[G.MODEL][G.STAT][stat][G.PAR]['k_bra_lb']

            elif field_name == G.KVAL_BRASSEUR_UPBOU:

                gust = data[G.MODEL][G.STAT][stat][G.PAR]['zv_bra_ub']
                kval = data[G.MODEL][G.STAT][stat][G.PAR]['k_bra_ub']

            elif field_name == G.MEAN_WIND:
                
                field = data[G.MODEL][G.STAT][stat][G.PAR]['zvp10']


            else:
                raise ValueError('Unknown user input "' + field_name + '" in list i_gust_fields!')
    
        
            # Aggregate to hourly values
            hourly_values = np.full((nhrs),np.nan)

            if field_name in G.FIELDS_KVAL: 

                # find index of maximum brasseur gust and store k value
                for hr in range(0,nhrs):
                    inds = hr_inds[hr]
                    # find maximum gust ind
                    hr_max_gust_ind = np.argmax(gust[inds])
                    hourly_values[hr] = kval[inds][hr_max_gust_ind]

            elif field_name in G.FIELDS_GUST: 

                # calc and save model max gust
                for hr in range(0,nhrs):
                    inds = hr_inds[hr]
                    # find maximum gust
                    hourly_values[hr] =np.max(gust[inds])
 
            else: # any other field

                # calc and save model mean hourly value
                for hr in range(0,nhrs):
                    inds = hr_inds[hr]
                    # calculate hourly mean value
                    hourly_values[hr] = np.mean(field[inds])
        
            # save in data
            data[G.MODEL][G.STAT][stat][G.FIELDS][field_name] = hourly_values

    return(data)
    




def calc_scores(data, i_scores):

    """
    Calculate scores for the hourly gusts in data

    INPUT
    data:           dictionary containing [G.MODEL][G.FIELDS] (with calcualted gusts)
    i_scores:       list containing string of scores to calculate.
                    for options see globals.py section 'scores':

    OUTPUT
    data:           added SCORE entry to MODEL according to i_scores
    """
    # loop through all stations
    for stat in data[G.STAT_NAMES]:

        # add score dictionary entry
        data[G.MODEL][G.STAT][stat][G.SCORE] = {}
    
        gust_methods = list(data[G.MODEL][G.STAT][stat][G.FIELDS].keys())
        #gust_methods = [gm for gm in gust_methods_inp if gm in G.FIELDS_GUST]
        #if len(gust_methods_inp) != len(gust_methods):
        #    raise ValueError('Non-gust field in score calculation!')

        # vector of observed gusts
        gust_obs = data[G.OBS][G.STAT][stat][G.PAR]['VMAX_10M1'].values
        # vector of observed mean winds
        wind_obs = data[G.OBS][G.STAT][stat][G.PAR]['FF_10M'].values

        for gust_method in gust_methods:

            # add entry for gust method in score dictionary
            data[G.MODEL][G.STAT][stat][G.SCORE][gust_method] = {}

            # vector of simulated gusts
            gust_mod = data[G.MODEL][G.STAT][stat][G.FIELDS][gust_method]

            if gust_method in G.FIELDS_GUST:
                eval_field = gust_obs
            else:
                eval_field = wind_obs
    
            for score_name in i_scores:
                
                # Vector scores
                if score_name == G.SCORE_ME:
                    score = gust_mod - eval_field
                elif score_name == G.SCORE_AE:
                    score = np.abs(gust_mod - eval_field)

                # scalar scores
                elif score_name == G.SCORE_MAE:
                    score = metrics.mean_absolute_error(eval_field, gust_mod)

                # add score to gust method dictionary
                data[G.MODEL][G.STAT][stat][G.SCORE][gust_method][score_name] = score
        

    return(data)








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

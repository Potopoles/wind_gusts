import numpy as np
import pandas as pd
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
                        
                    elif field_name in G.FIELDS_BRA_GUST:
                        if field_name == G.GUST_BRASSEUR_ESTIM:
                            gust_field_name = 'zv_bra_es'
                        elif field_name == G.GUST_BRASSEUR_LOBOU:
                            gust_field_name = 'zv_bra_lb'
                        elif field_name == G.GUST_BRASSEUR_UPBOU:
                            gust_field_name = 'zv_bra_ub'
            
                        gust = curr_raw[gust_field_name]

                    # resample to maximum hourly value
                    field = gust.resample('H').max()
                    field.columns = [field_name]

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
                    values = curr_raw[k_field_name][maxid].values
                    field = pd.DataFrame(values, index=maxid.index.levels[1], columns=[field_name]).astype(int)

                elif field_name == G.MEAN_WIND:
                    
                    field = curr_raw['zvp10']
                    field = field.resample('H').mean()

                else:
                    raise ValueError('Unknown user input "' + field_name + '" in list i_gust_fields!')
        
                all_fields.append(field)
            
            all_fields = pd.concat(all_fields, axis=1)

            # save in data
            data[G.MODEL][G.STAT][stat][G.FIELDS][lm_run] = all_fields

    return(data)
    

def join_model_and_obs(data):

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

    data[G.BOTH] = {G.STAT:{}}

    # loop through all stations
    for stat in data[G.STAT_NAMES]:

        data[G.BOTH][G.STAT][stat] = {}

        for lm_run in list(data[G.MODEL][G.STAT][stat][G.FIELDS].keys()):
            print(lm_run)
            quit()
            
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
                        
                    elif field_name in G.FIELDS_BRA_GUST:
                        if field_name == G.GUST_BRASSEUR_ESTIM:
                            gust_field_name = 'zv_bra_es'
                        elif field_name == G.GUST_BRASSEUR_LOBOU:
                            gust_field_name = 'zv_bra_lb'
                        elif field_name == G.GUST_BRASSEUR_UPBOU:
                            gust_field_name = 'zv_bra_ub'
            
                        gust = curr_raw[gust_field_name]

                    # resample to maximum hourly value
                    field = gust.resample('H').max()
                    field.columns = [field_name]

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
                    values = curr_raw[k_field_name][maxid].values
                    field = pd.DataFrame(values, index=maxid.index.levels[1], columns=[field_name]).astype(int)

                elif field_name == G.MEAN_WIND:
                    
                    field = curr_raw['zvp10']
                    field = field.resample('H').mean()

                else:
                    raise ValueError('Unknown user input "' + field_name + '" in list i_gust_fields!')
        
                all_fields.append(field)
            
            all_fields = pd.concat(all_fields, axis=1)

            # save in data
            data[G.MODEL][G.STAT][stat][G.FIELDS][lm_run] = all_fields

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

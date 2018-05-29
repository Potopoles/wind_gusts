import numpy as np
import globals as G


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
    
                gust = data[G.MODEL][G.STAT][stat][G.PAR]['zv_bra']

            elif method == G.GUST_BRASSEUR_K_VAL:

                gust = data[G.MODEL][G.STAT][stat][G.PAR]['zv_bra']
                kval = data[G.MODEL][G.STAT][stat][G.PAR]['k_bra']

            else:
                raise ValueError('Unknown user input "' + method + '" in list i_gust_fields!')
    
        
            # Aggregate to hourly values
            hr_max_gusts = np.full((nhrs),np.nan)

            if method == G.GUST_BRASSEUR_K_VAL:

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
    



def calc_scores(data, i_scores):

    """
    Calculate various scores according to various methods

    INPUT
    data:           dictionary containing data (with calcualted gusts)
    i_scores:       list containing string of scores to calculate.
                    for options see globals.py section 'scores':

    OUTPUT
    data:           added SCORE entry to MODEL according to i_scores
    """
    # calculate scores
    # error fields
    mod_err_gust = np.full((nhrs,nstat,4),np.nan)
    abs_err_gust = np.full((nhrs,nstat,4),np.nan)
    for mi in i_methods:
        mod_err_gust[:,:,mi-1] = mod_gust[:,:,mi-1] - obs_gust
    abs_err_gust = np.abs(mod_err_gust)


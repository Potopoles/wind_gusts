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
    data:           add GUSTS entry to MODEL according to i_gust_fields
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
        print(stat)

        # add gust dictionary entry
        data[G.MODEL][G.STAT][stat][G.GUST] = {}
    
        for method in i_gust_fields:
            #print(method)
    
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

            else:
                raise ValueError('Unknown user input in list "i_gust_fields"!')
    
        
            # calc and save model max gust
            hr_max_gusts = np.full((nhrs),np.nan)
            for hr in range(0,nhrs):
                inds = hr_inds[hr]
                # find maximum gust
                hr_max_gust = np.max(gust[inds])
                hr_max_gusts[hr] = hr_max_gust
    
            # save in data
            data[G.MODEL][G.STAT][stat][G.GUST][method] = hr_max_gusts
            #print(data[G.MODEL][G.STAT][stat][G.GUST][method])
    

def calc_scores(data, i_gust_fields):

    """
    Calculate gusts according to various methods

    INPUT
    data:           dictionary containing data
    i_gust_fields:  list containing string of gust method names:
                    for options see globals.py section 'Gust methods':

    OUTPUT
    data:           add GUSTS entry to MODEL according to i_gust_fields
    """
    # calculate scores
    # error fields
    mod_err_gust = np.full((nhrs,nstat,4),np.nan)
    abs_err_gust = np.full((nhrs,nstat,4),np.nan)
    for mi in i_methods:
        mod_err_gust[:,:,mi-1] = mod_gust[:,:,mi-1] - obs_gust
    abs_err_gust = np.abs(mod_err_gust)


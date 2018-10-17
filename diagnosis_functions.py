

def load_var_at_station_from_nc(nc_path, var_name, sel_stat):
    # GET STATION i AND j AND fort_stat INDEX
    station_file = "/users/heimc/stations/all_stations.lst"
    station_list = np.genfromtxt(station_file, skip_header=1, dtype=np.str)
    headers = station_list[0,:]
    station_list = station_list[1:,:]
    #print(headers)
    sel_stat_ind = station_list[:,0] == sel_stat
    #print(station_list[sel_stat_ind,:])
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

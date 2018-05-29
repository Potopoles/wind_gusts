###### Generic keys in data dictionary
# 1st level: key to model data
# data[MODEL]
MODEL = 'model'
# 1st level: key to obs data
# data[OBS]
OBS = 'obs'
# 1st level: key to names of stations
# data[STAT_NAMES]
STAT_NAMES = 'station_names'
# 2nd level: key in MODEL and OBS leading to station dictionary
# data[MODEL/OBS][STAT]
STAT = 'stations'
# 2nd level: key in MODEL and OBS showing the names of parameters contained for each station
# data[MODEL/OBS][PAR_NAMES]
PAR_NAMES = 'param_names'
# 2nd level: key in MODEL and OBS containing datetime objects for each time step
# data[MODEL/OBS][DTS]
DTS = 'dts'
# 3rd level: key in STAT leading to observed/modeled raw fields 
# data[MODEL/OBS][STAT][PAR]
PAR = 'params'
# 3rd level MODEL:  key to hourly gust data
# data[MODEL][STAT][GUST]
GUST = 'gusts'



###### Gust methods
GUST_MIX_COEF_LINEAR = 'gust_mix_coef_linear'
GUST_MIX_COEF_NONLIN = 'gust_mix_coef_nonlin'
GUST_BRASSEUR_ESTIM  = 'gust_brassuer_estim'

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
# 3rd level: key in STAT leading to meta information about station (e.g. tags)
# data[MODEL/OBS][STAT][STAT_META]
STAT_META = 'station_meta'
# 3rd level: key in STAT leading to observed/modeled raw fields 
# data[MODEL/OBS][STAT][PAR]
PAR = 'params'
# 3rd level MODEL:  key to hourly gust data
# data[MODEL][STAT][FIELDS]
FIELDS = 'fields'
# 3rd level MODEL:  key to hourly scores data
# data[MODEL][STAT][SCORE]
SCORE = 'scores'



###### Model Fields
GUST_MIX_COEF_LINEAR = 'gust_mix_coef_linear'
GUST_MIX_COEF_NONLIN = 'gust_mix_coef_nonlin'
GUST_BRASSEUR_ESTIM  = 'gust_brasseur_estim'
GUST_BRASSEUR_LOBOU  = 'gust_brasseur_lobou'
GUST_BRASSEUR_UPBOU  = 'gust_brasseur_upbou'
FIELDS_GUST = [GUST_MIX_COEF_LINEAR,
               GUST_MIX_COEF_NONLIN,
               GUST_BRASSEUR_ESTIM,
               GUST_BRASSEUR_LOBOU,
               GUST_BRASSEUR_UPBOU]
KVAL_BRASSEUR_ESTIM  = 'kval_brasseur_estim'
KVAL_BRASSEUR_LOBOU  = 'kval_brasseur_lobou'
KVAL_BRASSEUR_UPBOU  = 'kval_brasseur_upbou'
FIELDS_KVAL = [KVAL_BRASSEUR_ESTIM,
               KVAL_BRASSEUR_LOBOU,
               KVAL_BRASSEUR_UPBOU]
MEAN_WIND = 'mean_wind'



###### Scores
# vector scores
# model error
SCORE_ME = 'score_me'
# absolute error
SCORE_AE = 'score_ae'
# scalar scores
# mean absolute error
SCORE_MAE = 'score_mae'

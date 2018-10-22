###### Generic keys in data dictionary
# 1st level: history of functions applied to data
# data[HIST]
HIST = 'hist'
# 1st level: key to model data
# data[MODEL]
#MODEL = 'model'
# 1st level: key to obs data
# data[OBS]
OBS = 'obs'
## 1st level: key to combined data
## data[BOTH]
#BOTH = 'both'
## 1st level:  key to hourly scores data
## data[SCORE]
#SCORE = 'scores'
# 1st level: key to names of stations
# data[STAT_NAMES]
STAT_NAMES = 'station_names'
# 1st level: key leading to meta information about station (e.g. tags)
# data[STAT_META]
STAT_META = 'station_meta'
# 2nd level: key in MODEL and OBS leading to station dictionary
# data[MODEL/OBS][STAT]
STAT = 'stations'
## 2nd level: key in BOTH containing arrays for all stations combined 
## data[BOTH][ALL_STAT]
#ALL_STAT = 'all_stations'
## 3rd level: key in STAT leading to raw (time step) model output fields 
## data[MODEL][STAT][RAW]
#RAW = 'raw'
## 3rd level MODEL:  key to hourly gust data
## data[MODEL][STAT][FIELDS]
#FIELDS = 'fields'



###### Model Fields
GUST_MIX_COEF_LINEAR = 'gust_mix_coef_linear'
GUST_MIX_COEF_NONLIN = 'gust_mix_coef_nonlin'
GUST_BRASSEUR_ESTIM  = 'gust_brasseur_estim'
GUST_BRASSEUR_LOBOU  = 'gust_brasseur_lobou'
GUST_BRASSEUR_UPBOU  = 'gust_brasseur_upbou'
GUST_ICON            = 'gust_icon'
KVAL_BRASSEUR_ESTIM  = 'kval_brasseur_estim'
KVAL_BRASSEUR_LOBOU  = 'kval_brasseur_lobou'
KVAL_BRASSEUR_UPBOU  = 'kval_brasseur_upbou'
MODEL_MEAN_WIND      = 'model_mean_wind'
MEAN_WIND_INST      = 'mean_wind_inst'
FIELDS_GUST = [GUST_MIX_COEF_LINEAR,
               GUST_MIX_COEF_NONLIN,
               GUST_BRASSEUR_ESTIM,
               GUST_BRASSEUR_LOBOU,
               GUST_BRASSEUR_UPBOU,
               GUST_ICON]
FIELDS_BRA_GUST =  [GUST_BRASSEUR_ESTIM,
                    GUST_BRASSEUR_LOBOU,
                    GUST_BRASSEUR_UPBOU]
FIELDS_BRA_KVAL = [KVAL_BRASSEUR_ESTIM,
                   KVAL_BRASSEUR_LOBOU,
                   KVAL_BRASSEUR_UPBOU]

###### Obs Fields
OBS_GUST_SPEED = 'obs_gust_speed'
OBS_MEAN_WIND  = 'obs_mean_wind'
OBS_WIND_DIR   = 'obs_wind_dir'



###### Scores
# vector scores
# model error
SCORE_ME = 'score_me'
# absolute error
SCORE_AE = 'score_ae'
# scalar scores
# mean absolute error
SCORE_MAE = 'score_mae'

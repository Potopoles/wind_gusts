import numpy as np

####################################################################################################
####################################################################################################
###                                 READJUST
####################################################################################################
####################################################################################################









####################################################################################################
####################################################################################################
###                                 STATISTICAL
####################################################################################################
####################################################################################################


def stat_calculate_gust(mode, features, alphas, zvp10_unsc):
    if mode == 'mean':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10']
    elif mode == 'mean_mean2':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['zvp10']**2
    elif mode == 'mean_tke':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['tkel1']
    elif mode == 'mean_height':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['hsurf']
    elif mode == 'mean_gustbra':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['zv_bra_es']
    elif mode == 'mean_gustbra_tke':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['zv_bra_es'] \
                + alphas[3]*features['tkel1']
    elif mode == 'mean_gustbra_height':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['zv_bra_es'] \
                + alphas[3]*features['hsurf']
    elif mode == 'mean_gustbra_dvl3v10':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['zv_bra_es'] \
                + alphas[3]*features['dvl3v10']
    elif mode == 'mean_zbra':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['zbra']
    elif mode == 'mean_dvl3v10':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['dvl3v10']
    elif mode == 'mean_icon':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['icon_gust']
    elif mode == 'mean_gustbra_icon':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['zv_bra_es'] + \
                + alphas[3]*features['icon_gust']
    elif mode == 'mean_gustbra_mean2':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['zv_bra_es'] \
                + alphas[3]*features['zvp10']**2
    elif mode == 'mean_gustbra_mean2_icon':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['zv_bra_es'] \
                + alphas[3]*features['zvp10']**2 + alphas[4]*features['icon_gust']
    elif mode == 'mean_gustbra_mean2_zbra':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['zv_bra_es'] \
                + alphas[3]*features['zvp10']**2 + alphas[4]*features['zbra']
    else:
        raise ValueError('wrong mode')
    return(gust)

def stat_combine_features(mode, features, zvp10_unsc):
    trained = {}
    if mode == 'mean':
        trained[1] = {'feat':'zvp10','power':1}
    elif mode == 'mean_mean2':
        trained[1] = {'feat':'zvp10','power':1}
        trained[2] = {'feat':'zvp10','power':2}
    elif mode == 'mean_tke':
        trained[1] = {'feat':'zvp10','power':1}
        trained[2] = {'feat':'tkel1','power':1}
    elif mode == 'mean_height':
        trained[1] = {'feat':'zvp10','power':1}
        trained[2] = {'feat':'hsurf','power':1}
    elif mode == 'mean_gustbra':
        trained[1] = {'feat':'zvp10','power':1}
        trained[2] = {'feat':'zv_bra_es','power':1}
    elif mode == 'mean_gustbra_tke':
        trained[1] = {'feat':'zvp10','power':1}
        trained[2] = {'feat':'zv_bra_es','power':1}
        trained[3] = {'feat':'tkel1','power':1}
    elif mode == 'mean_gustbra_height':
        trained[1] = {'feat':'zvp10','power':1}
        trained[2] = {'feat':'zv_bra_es','power':1}
        trained[3] = {'feat':'hsurf','power':1}
    elif mode == 'mean_gustbra_dvl3v10':
        trained[1] = {'feat':'zvp10','power':1}
        trained[2] = {'feat':'zv_bra_es','power':1}
        trained[3] = {'feat':'dvl3v10','power':1}
    elif mode == 'mean_zbra':
        trained[1] = {'feat':'zvp10','power':1}
        trained[2] = {'feat':'zbra','power':1}
    elif mode == 'mean_dvl3v10':
        trained[1] = {'feat':'zvp10','power':1}
        trained[2] = {'feat':'dvl3v10','power':1}
    elif mode == 'mean_icon':
        trained[1] = {'feat':'zvp10','power':1}
        trained[2] = {'feat':'icon_gust','power':1}
    elif mode == 'mean_gustbra_icon':
        trained[1] = {'feat':'zvp10','power':1}
        trained[2] = {'feat':'zv_bra_es','power':1}
        trained[3] = {'feat':'icon_gust','power':1}
    elif mode == 'mean_gustbra_mean2':
        trained[1] = {'feat':'zvp10','power':1}
        trained[2] = {'feat':'zv_bra_es','power':1}
        trained[3] = {'feat':'zvp10','power':2}
    elif mode == 'mean_gustbra_mean2_icon':
        trained[1] = {'feat':'zvp10','power':1}
        trained[2] = {'feat':'zv_bra_es','power':1}
        trained[3] = {'feat':'zvp10','power':2}
        trained[4] = {'feat':'icon_gust','power':1}
    elif mode == 'mean_gustbra_mean2_zbra':
        trained[1] = {'feat':'zvp10','power':1}
        trained[2] = {'feat':'zbra','power':1}
        trained[3] = {'feat':'zvp10','power':2}
        trained[4] = {'feat':'k_bra_es','power':1}

    nfeat = len(trained) + 1
    X = np.full((zvp10_unsc.shape[0],nfeat), np.nan)
    X[:,0] = 1
    c = 1
    for key,value in trained.items():
        X[:,c] = features[value['feat']]**value['power']
        c += 1
    return(X, trained)





####################################################################################################
####################################################################################################
###                                 BRASSEUR ESTIMATE
####################################################################################################
####################################################################################################

def braes_feature_matrix(mode, gust_est_max, kheight_est_max,
                                height_max, model_mean_max):
    if mode == 'gust':
        X = np.zeros((len(gust_est_max), 1))
        X[:,0] = gust_est_max
    elif mode == 'gust_kheight':
        X = np.zeros((len(gust_est_max), 2))
        X[:,0] = gust_est_max
        X[:,1] = kheight_est_max
    elif mode == 'gust_height':
        X = np.zeros((len(gust_est_max), 2))
        X[:,0] = gust_est_max
        X[:,1] = height_max
    elif mode == 'gust_mean':
        X = np.zeros((len(gust_est_max), 2))
        X[:,0] = gust_est_max
        X[:,1] = model_mean_max
    elif mode == 'gust_mean_mean2':
        X = np.zeros((len(gust_est_max), 3))
        X[:,0] = gust_est_max
        X[:,1] = model_mean_max
        X[:,2] = model_mean_max**2
    elif mode == 'gust_mean_mean2_height':
        X = np.zeros((len(gust_est_max), 4))
        X[:,0] = gust_est_max
        X[:,1] = model_mean_max
        X[:,2] = model_mean_max**2
        X[:,3] = height_max
    elif mode == 'gust_mean_mean2_mean3':
        X = np.zeros((len(gust_est_max), 4))
        X[:,0] = gust_est_max
        X[:,1] = model_mean_max
        X[:,2] = model_mean_max**2
        X[:,3] = model_mean_max**3
    elif mode == 'gust_mean_mean2_mean3_height':
        X = np.zeros((len(gust_est_max), 5))
        X[:,0] = gust_est_max
        X[:,1] = model_mean_max
        X[:,2] = model_mean_max**2
        X[:,3] = model_mean_max**3
        X[:,4] = height_max
    elif mode == 'gust_mean_kheight':
        X = np.zeros((len(gust_est_max), 3))
        X[:,0] = gust_est_max
        X[:,1] = model_mean_max
        X[:,2] = kheight_est_max
    elif mode == 'gust_mean_height':
        X = np.zeros((len(gust_est_max), 3))
        X[:,0] = gust_est_max
        X[:,1] = model_mean_max
        X[:,2] = height_max
    elif mode == 'gust_mean_height_mean2_kheight':
        X = np.zeros((len(gust_est_max), 5))
        X[:,0] = gust_est_max
        X[:,1] = model_mean_max
        X[:,2] = height_max
        X[:,3] = model_mean_max**2
        X[:,4] = kheight_est_max
    else:
        raise ValueError('wrong mode')
    X = np.append(np.ones( (X.shape[0],1) ), X, axis=1)

    return(X)








####################################################################################################
####################################################################################################
###                                 BRASSEUR LOWER BOUND
####################################################################################################
####################################################################################################

def bralb_feature_matrix(mode, gust_lb_max, kheight_lb_max,
                                height_max, model_mean_max):
    if mode == 'gust':
        X = np.zeros((len(gust_lb_max), 1))
        X[:,0] = gust_lb_max
    elif mode == 'gust_gust2':
        X = np.zeros((len(gust_lb_max), 2))
        X[:,0] = gust_lb_max
        X[:,1] = gust_lb_max**2
    elif mode == 'gust_kheight':
        X = np.zeros((len(gust_lb_max), 2))
        X[:,0] = gust_lb_max
        X[:,1] = kheight_lb_max
    elif mode == 'gust_height':
        X = np.zeros((len(gust_lb_max), 2))
        X[:,0] = gust_lb_max
        X[:,1] = height_max
    elif mode == 'gust_mean':
        X = np.zeros((len(gust_lb_max), 2))
        X[:,0] = gust_lb_max
        X[:,1] = model_mean_max
    elif mode == 'gust_mean_mean2':
        X = np.zeros((len(gust_lb_max), 3))
        X[:,0] = gust_lb_max
        X[:,1] = model_mean_max
        X[:,2] = model_mean_max**2
    elif mode == 'gust_mean_kheight':
        X = np.zeros((len(gust_lb_max), 3))
        X[:,0] = gust_lb_max
        X[:,1] = model_mean_max
        X[:,2] = kheight_lb_max
    elif mode == 'gust_mean_height':
        X = np.zeros((len(gust_lb_max), 3))
        X[:,0] = gust_lb_max
        X[:,1] = model_mean_max
        X[:,2] = height_max
    elif mode == 'gust_mean_height_mean2':
        X = np.zeros((len(gust_lb_max), 4))
        X[:,0] = gust_lb_max
        X[:,1] = model_mean_max
        X[:,2] = height_max
        X[:,3] = model_mean_max**2
    elif mode == 'gust_mean_height_mean2_kheight':
        X = np.zeros((len(gust_lb_max), 5))
        X[:,0] = gust_lb_max
        X[:,1] = model_mean_max
        X[:,2] = height_max
        X[:,3] = model_mean_max**2
        X[:,4] = kheight_lb_max
    else:
        raise ValueError('wrong mode')
    X = np.append(np.ones( (X.shape[0],1) ), X, axis=1)

    return(X)








####################################################################################################
####################################################################################################
###                                            ICON
####################################################################################################
####################################################################################################

def icon_feature_matrix(mode, gust_lb_max, kheight_lb_max,
                                height_max, model_mean_max):
    if mode == 'gust':
        X = np.zeros((len(gust_ico_max), 1))
        X[:,0] = gust_ico_max
    elif mode == 'gust_mean2':
        X = np.zeros((len(gust_ico_max), 2))
        X[:,0] = gust_ico_max
        X[:,1] = model_mean_max**2
    elif mode == 'gust_gust2':
        X = np.zeros((len(gust_ico_max), 2))
        X[:,0] = gust_ico_max
        X[:,1] = gust_ico_max**2
    elif mode == 'gust_gust2_height':
        X = np.zeros((len(gust_ico_max), 3))
        X[:,0] = gust_ico_max
        X[:,1] = gust_ico_max**2
        X[:,2] = height_max
    elif mode == 'gust_gust2_height_mean2':
        X = np.zeros((len(gust_ico_max), 4))
        X[:,0] = gust_ico_max
        X[:,1] = gust_ico_max**2
        X[:,2] = height_max
        X[:,3] = model_mean_max**2
    elif mode == 'gust_gust2_height_mean2_mean':
        X = np.zeros((len(gust_ico_max), 5))
        X[:,0] = gust_ico_max
        X[:,1] = gust_ico_max**2
        X[:,2] = height_max
        X[:,3] = model_mean_max**2
        X[:,4] = model_mean_max
    elif mode == 'gust_gust2_height_mean2_mean_dvl3v10':
        X = np.zeros((len(gust_ico_max), 6))
        X[:,0] = gust_ico_max
        X[:,1] = gust_ico_max**2
        X[:,2] = height_max
        X[:,3] = model_mean_max**2
        X[:,4] = model_mean_max
        X[:,5] = dvl3v10_max
    elif mode == 'gust_gust2_height_mean2_mean_dvl3v10_tkel1':
        X = np.zeros((len(gust_ico_max), 7))
        X[:,0] = gust_ico_max
        X[:,1] = gust_ico_max**2
        X[:,2] = height_max
        X[:,3] = model_mean_max**2
        X[:,4] = model_mean_max
        X[:,5] = dvl3v10_max
        X[:,6] = tkel1_max
    elif mode == 'gust_kheight':
        X = np.zeros((len(gust_ico_max), 2))
        X[:,0] = gust_ico_max
        X[:,1] = kheight_est_max
    elif mode == 'gust_height':
        X = np.zeros((len(gust_ico_max), 2))
        X[:,0] = gust_ico_max
        X[:,1] = height_max
    elif mode == 'gust_dvl3v10':
        X = np.zeros((len(gust_ico_max), 2))
        X[:,0] = gust_ico_max
        X[:,1] = dvl3v10_max
    elif mode == 'gust_dvl3v10_height':
        X = np.zeros((len(gust_ico_max), 3))
        X[:,0] = gust_ico_max
        X[:,1] = dvl3v10_max
        X[:,2] = height_max
    elif mode == 'gust_dvl3v10_height_mean2':
        X = np.zeros((len(gust_ico_max), 4))
        X[:,0] = gust_ico_max
        X[:,1] = dvl3v10_max
        X[:,2] = height_max
        X[:,3] = model_mean_max**2
    elif mode == 'gust_dvl3v10_height_mean2_gust2':
        X = np.zeros((len(gust_ico_max), 5))
        X[:,0] = gust_ico_max
        X[:,1] = dvl3v10_max
        X[:,2] = height_max
        X[:,3] = model_mean_max**2
        X[:,4] = gust_ico_max**2
    elif mode == 'gust_braest':
        X = np.zeros((len(gust_ico_max), 2))
        X[:,0] = gust_ico_max
        X[:,1] = gust_bra_est
    elif mode == 'gust_bralb':
        X = np.zeros((len(gust_ico_max), 2))
        X[:,0] = gust_ico_max
        X[:,1] = gust_bra_lb
    elif mode == 'gust_braest_kheight':
        X = np.zeros((len(gust_ico_max), 3))
        X[:,0] = gust_ico_max
        X[:,1] = gust_bra_est
        X[:,2] = kheight_est_max
    elif mode == 'gust_braest_height':
        X = np.zeros((len(gust_ico_max), 3))
        X[:,0] = gust_ico_max
        X[:,1] = gust_bra_est
        X[:,2] = height_max
    elif mode == 'gust_braest_height_gust2':
        X = np.zeros((len(gust_ico_max), 4))
        X[:,0] = gust_ico_max
        X[:,1] = gust_bra_est
        X[:,2] = height_max
        X[:,3] = gust_ico_max**2
    elif mode == 'gust_braest_kheight_height':
        X = np.zeros((len(gust_ico_max), 4))
        X[:,0] = gust_ico_max
        X[:,1] = gust_bra_est
        X[:,2] = kheight_est_max
        X[:,3] = height_max
    else:
        raise ValueError('wrong mode')
    X = np.append(np.ones( (X.shape[0],1) ), X, axis=1)

    return(X)


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
    elif mode == 'mean_mean2_gustbra_dvl3v10':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['zvp10']**2 \
                + alphas[3]*features['zv_bra_es'] + alphas[4]*features['dvl3v10']
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
    elif mode == 'mean_mean2_gustbra_dvl3v10':
        trained[1] = {'feat':'zvp10','power':1}
        trained[2] = {'feat':'zvp10','power':2}
        trained[3] = {'feat':'zv_bra_es','power':1}
        trained[4] = {'feat':'dvl3v10','power':1}
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


def braes_feature_matrix_timestep(mode, gust_est, kheight_est,
                                height, model_mean):
    if mode == 'gust':
        X = np.zeros((gust_est.shape[0], gust_est.shape[1], 1))
        X[:,:,0] = gust_est
    elif mode == 'gust_kheight':
        X = np.zeros((gust_est.shape[0], gust_est.shape[1], 2))
        X[:,:,0] = gust_est
        X[:,:,1] = kheight_est
    elif mode == 'gust_height':
        X = np.zeros((gust_est.shape[0], gust_est.shape[1], 2))
        X[:,:,0] = gust_est
        X[:,:,1] = height
    elif mode == 'gust_mean':
        X = np.zeros((gust_est.shape[0], gust_est.shape[1], 2))
        X[:,:,0] = gust_est
        X[:,:,1] = model_mean
    elif mode == 'gust_mean_mean2':
        X = np.zeros((gust_est.shape[0], gust_est.shape[1], 3))
        X[:,:,0] = gust_est
        X[:,:,1] = model_mean
        X[:,:,2] = model_mean**2
    elif mode == 'gust_mean_mean2_height':
        X = np.zeros((gust_est.shape[0], gust_est.shape[1], 4))
        X[:,:,0] = gust_est
        X[:,:,1] = model_mean
        X[:,:,2] = model_mean**2
        X[:,:,3] = height
    elif mode == 'gust_mean_mean2_mean3':
        X = np.zeros((gust_est.shape[0], gust_est.shape[1], 4))
        X[:,:,0] = gust_est
        X[:,:,1] = model_mean
        X[:,:,2] = model_mean**2
        X[:,:,3] = model_mean**3
    elif mode == 'gust_mean_mean2_mean3_height':
        X = np.zeros((gust_est.shape[0], gust_est.shape[1], 5))
        X[:,:,0] = gust_est
        X[:,:,1] = model_mean
        X[:,:,2] = model_mean**2
        X[:,:,3] = model_mean**3
        X[:,:,4] = height
    elif mode == 'gust_mean_kheight':
        X = np.zeros((gust_est.shape[0], gust_est.shape[1], 3))
        X[:,:,0] = gust_est
        X[:,:,1] = model_mean
        X[:,:,2] = kheight_est
    elif mode == 'gust_mean_height':
        X = np.zeros((gust_est.shape[0], gust_est.shape[1], 3))
        X[:,:,0] = gust_est
        X[:,:,1] = model_mean
        X[:,:,2] = height
    elif mode == 'gust_mean_height_mean2_kheight':
        X = np.zeros((gust_est.shape[0], gust_est.shape[1], 5))
        X[:,:,0] = gust_est
        X[:,:,1] = model_mean
        X[:,:,2] = height
        X[:,:,3] = model_mean**2
        X[:,:,4] = kheight_est
    else:
        raise ValueError('wrong mode')
    X = np.append(np.ones( (X.shape[0],X.shape[1],1) ), X, axis=2)

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


def bralb_feature_matrix_timestep(mode, gust_lb, kheight_lb,
                                height, model_mean):
    if mode == 'gust':
        X = np.zeros((gust_lb.shape[0], gust_lb.shape[1], 1))
        X[:,:,0] = gust_lb
    elif mode == 'gust_gust2':
        X = np.zeros((gust_lb.shape[0], gust_lb.shape[1], 2))
        X[:,:,0] = gust_lb
        X[:,:,1] = gust_lb**2
    elif mode == 'gust_kheight':
        X = np.zeros((gust_lb.shape[0], gust_lb.shape[1], 2))
        X[:,:,0] = gust_lb
        X[:,:,1] = kheight_lb
    elif mode == 'gust_height':
        X = np.zeros((gust_lb.shape[0], gust_lb.shape[1], 2))
        X[:,:,0] = gust_lb
        X[:,:,1] = height
    elif mode == 'gust_mean':
        X = np.zeros((gust_lb.shape[0], gust_lb.shape[1], 2))
        X[:,:,0] = gust_lb
        X[:,:,1] = model_mean
    elif mode == 'gust_mean_mean2':
        X = np.zeros((gust_lb.shape[0], gust_lb.shape[1], 3))
        X[:,:,0] = gust_lb
        X[:,:,1] = model_mean
        X[:,:,2] = model_mean**2
    elif mode == 'gust_mean_kheight':
        X = np.zeros((gust_lb.shape[0], gust_lb.shape[1], 3))
        X[:,:,0] = gust_lb
        X[:,:,1] = model_mean
        X[:,:,2] = kheight_lb
    elif mode == 'gust_mean_height':
        X = np.zeros((gust_lb.shape[0], gust_lb.shape[1], 3))
        X[:,:,0] = gust_lb
        X[:,:,1] = model_mean
        X[:,:,2] = height
    elif mode == 'gust_mean_height_mean2':
        X = np.zeros((gust_lb.shape[0], gust_lb.shape[1], 4))
        X[:,:,0] = gust_lb
        X[:,:,1] = model_mean
        X[:,:,2] = height
        X[:,:,3] = model_mean**2
    elif mode == 'gust_mean_height_mean2_kheight':
        X = np.zeros((gust_lb.shape[0], gust_lb.shape[1], 5))
        X[:,:,0] = gust_lb
        X[:,:,1] = model_mean
        X[:,:,2] = height
        X[:,:,3] = model_mean**2
        X[:,:,4] = kheight_lb
    else:
        raise ValueError('wrong mode')
    X = np.append(np.ones( (X.shape[0],X.shape[1],1) ), X, axis=2)

    return(X)






####################################################################################################
####################################################################################################
###                                            ICON
####################################################################################################
####################################################################################################

def icon_feature_matrix(mode, gust_ico_max, height_max,
                                dvl3v10_max, model_mean_max,
                                tkel1_max):

    if mode == 'gust_mean':
        X = np.zeros((len(gust_ico_max), 2))
        X[:,0] = gust_ico_max
        X[:,1] = model_mean_max
    elif mode == 'gust_mean_mean2':
        X = np.zeros((len(gust_ico_max), 3))
        X[:,0] = gust_ico_max
        X[:,1] = model_mean_max
        X[:,2] = model_mean_max**2
    elif mode == 'gust_mean_height':
        X = np.zeros((len(gust_ico_max), 3))
        X[:,0] = gust_ico_max
        X[:,1] = model_mean_max
        X[:,2] = height_max
    elif mode == 'gust_mean_mean2_height':
        X = np.zeros((len(gust_ico_max), 4))
        X[:,0] = gust_ico_max
        X[:,1] = model_mean_max
        X[:,2] = model_mean_max**2
        X[:,3] = height_max
    elif mode == 'gust_mean_tkel1':
        X = np.zeros((len(gust_ico_max), 3))
        X[:,0] = gust_ico_max
        X[:,1] = gust_ico_max
        X[:,2] = tkel1_max
    elif mode == 'gust_mean_mean2_tkel1':
        X = np.zeros((len(gust_ico_max), 4))
        X[:,0] = gust_ico_max
        X[:,1] = model_mean_max
        X[:,2] = model_mean_max**2
        X[:,3] = tkel1_max
    elif mode == 'gust_mean_mean2_height_tkel1':
        X = np.zeros((len(gust_ico_max), 5))
        X[:,0] = gust_ico_max
        X[:,1] = model_mean_max
        X[:,2] = model_mean_max**2
        X[:,3] = height_max
        X[:,4] = tkel1_max
    elif mode == 'gust_mean_mean2_height_tkel1_dvl3v10':
        X = np.zeros((len(gust_ico_max), 6))
        X[:,0] = gust_ico_max
        X[:,1] = model_mean_max
        X[:,2] = model_mean_max**2
        X[:,3] = height_max
        X[:,4] = tkel1_max
        X[:,5] = dvl3v10_max
    elif mode == 'gust_mean_mean2_height_dvl3v10':
        X = np.zeros((len(gust_ico_max), 5))
        X[:,0] = gust_ico_max
        X[:,1] = model_mean_max
        X[:,2] = model_mean_max**2
        X[:,3] = height_max
        X[:,4] =dvl3v10_max 
    elif mode == 'gust_mean_mean2_tkel1_dvl3v10':
        X = np.zeros((len(gust_ico_max), 5))
        X[:,0] = gust_ico_max
        X[:,1] = model_mean_max
        X[:,2] = model_mean_max**2
        X[:,3] = tkel1_max
        X[:,4] =dvl3v10_max 
    else:
        raise ValueError('wrong mode')
    X = np.append(np.ones( (X.shape[0],1) ), X, axis=1)

    return(X)



def icon_feature_matrix_timestep(mode, gust_ico, height,
                                dvl3v10, model_mean,
                                tkel1):

    if mode == 'gust_mean':
        X = np.zeros((gust_ico.shape[0], gust_ico.shape[1], 2))
        X[:,:,0] = gust_ico
        X[:,:,1] = model_mean
    elif mode == 'gust_mean_mean2':
        X = np.zeros((gust_ico.shape[0], gust_ico.shape[1], 3))
        X[:,:,0] = gust_ico
        X[:,:,1] = model_mean
        X[:,:,2] = model_mean**2
    elif mode == 'gust_mean_height':
        X = np.zeros((gust_ico.shape[0], gust_ico.shape[1], 3))
        X[:,:,0] = gust_ico
        X[:,:,1] = model_mean
        X[:,:,2] = height
    elif mode == 'gust_mean_mean2_height':
        X = np.zeros((gust_ico.shape[0], gust_ico.shape[1], 4))
        X[:,:,0] = gust_ico
        X[:,:,1] = model_mean
        X[:,:,2] = model_mean**2
        X[:,:,3] = height
    elif mode == 'gust_mean_tkel1':
        X = np.zeros((gust_ico.shape[0], gust_ico.shape[1], 3))
        X[:,:,0] = gust_ico
        X[:,:,1] = gust_ico
        X[:,:,2] = tkel1
    elif mode == 'gust_mean_mean2_tkel1':
        X = np.zeros((gust_ico.shape[0], gust_ico.shape[1], 4))
        X[:,:,0] = gust_ico
        X[:,:,1] = model_mean
        X[:,:,2] = model_mean**2
        X[:,:,3] = tkel1
    elif mode == 'gust_mean_mean2_height_tkel1':
        X = np.zeros((gust_ico.shape[0], gust_ico.shape[1], 5))
        X[:,:,0] = gust_ico
        X[:,:,1] = model_mean
        X[:,:,2] = model_mean**2
        X[:,:,3] = height
        X[:,:,4] = tkel1
    elif mode == 'gust_mean_mean2_height_tkel1_dvl3v10':
        X = np.zeros((gust_ico.shape[0], gust_ico.shape[1], 6))
        X[:,:,0] = gust_ico
        X[:,:,1] = model_mean
        X[:,:,2] = model_mean**2
        X[:,:,3] = height
        X[:,:,4] = tkel1
        X[:,:,5] = dvl3v10
    elif mode == 'gust_mean_mean2_height_dvl3v10':
        X = np.zeros((gust_ico.shape[0], gust_ico.shape[1], 5))
        X[:,:,0] = gust_ico
        X[:,:,1] = model_mean
        X[:,:,2] = model_mean**2
        X[:,:,3] = height
        X[:,:,4] = dvl3v10 
    elif mode == 'gust_mean_mean2_tkel1_dvl3v10':
        X = np.zeros((gust_ico.shape[0], gust_ico.shape[1], 5))
        X[:,:,0] = gust_ico
        X[:,:,1] = model_mean
        X[:,:,2] = model_mean**2
        X[:,:,3] = tkel1
        X[:,:,4] = dvl3v10 
    else:
        raise ValueError('wrong mode')
    X = np.append(np.ones( (X.shape[0],X.shape[1],1) ), X, axis=2)

    return(X)


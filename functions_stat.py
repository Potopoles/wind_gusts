import numpy as np

def calculate_gust(mode, features, alphas, zvp10_unsc):
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
    elif mode == 'mean_kbra':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['k_bra_es']
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
    elif mode == 'mean_gustbra_mean2_kbra':
        gust = zvp10_unsc + alphas[0] + alphas[1]*features['zvp10'] + alphas[2]*features['zv_bra_es'] \
                + alphas[3]*features['zvp10']**2 + alphas[4]*features['k_bra_es']
    else:
        raise ValueError('wrong mode')
    return(gust)

def combine_features(mode, features, zvp10_unsc):
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
    elif mode == 'mean_kbra':
        trained[1] = {'feat':'zvp10','power':1}
        trained[2] = {'feat':'k_bra_es','power':1}
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
    elif mode == 'mean_gustbra_mean2_kbra':
        trained[1] = {'feat':'zvp10','power':1}
        trained[2] = {'feat':'zv_bra_es','power':1}
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

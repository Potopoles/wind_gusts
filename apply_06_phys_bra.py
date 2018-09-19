import numpy as np
import os
import copy
import matplotlib.pyplot as plt
import pickle
from functions import plot_error
import globals as G
from namelist_cases import Case_Namelist
import namelist_cases as nl

############ USER INPUT #############
train_case_index = nl.train_case_index
apply_case_index = nl.apply_case_index
CNtrain = Case_Namelist(train_case_index)
CNapply = Case_Namelist(apply_case_index)
# do not plot (0) show plot (1) save plot (2)
i_plot = nl.apply_i_plot
model_dt = nl.apply_model_dt
i_label = ''
#####################################

# create directories
if i_plot > 1 and not os.path.exists(CNapply.plot_path):
    os.mkdir(CNapply.plot_path)

# LOAD PARAMS
params = pickle.load( open(CNtrain.params_phys_bra_path, 'rb') )
#print(params)
#quit()

# LOAD FILES
data = pickle.load( open(CNapply.phys_bra_path, 'rb') )
model_mean_hr = data['model_mean_hr']
model_mean = data['model_mean']
gust_est = data['gust_est']
gust_lb = data['gust_lb']
kheight_est = data['kheight_est']
kheight_lb = data['kheight_lb']
#rho_est = data['rho_est']
#rho_lb = data['rho_lb']
#rho_surf = data['rho_surf']
obs_gust = data['obs_gust']
obs_mean = data['obs_mean']

# obs nan mask
obsmask = np.isnan(obs_gust)
obsmask[np.isnan(obs_mean)] = True

#mean_abs_error = np.abs(model_mean_hr - obs_mean)
#mean_rel_error = mean_abs_error/obs_mean
#errormask = mean_rel_error > max_mean_wind_error
## combine both
#obsmask[errormask] = True

obs_gust = obs_gust[~obsmask] 
obs_mean = obs_mean[~obsmask]
model_mean_hr = model_mean_hr[~obsmask]
model_mean = model_mean[~obsmask]
kheight_est = kheight_est[~obsmask]
gust_est = gust_est[~obsmask]
kheight_lb = kheight_lb[~obsmask]
gust_lb = gust_lb[~obsmask]
#rho_est = rho_est[~obsmask]
#rho_lb = rho_lb[~obsmask]
#rho_surf = rho_surf[~obsmask]


for mode in params.keys():
    print('#################################################################################')
    print('############################## ' + str(mode) + ' ################################')

    alphas = params[mode]

    # calc current time step gusts
    if mode == 'es':
        gust = gust_est - alphas[0]*kheight_est*(gust_est -  model_mean)
    if mode == 'lb':
        gust = gust_lb - alphas[0]*kheight_lb*(gust_lb -  model_mean)
    #elif mode == 'es_rho':
    #    gust = gust_est*rho_est/rho_surf - alphas[0]*kheight_est*\
    #                (gust_est*rho_est/rho_surf -  model_mean)
    #elif mode == 'lb_rho':
    #    gust = gust_lb*rho_lb/rho_surf - alphas[0]*kheight_lb*\
    #                (gust_lb*rho_lb/rho_surf -  model_mean)

    gust[gust < 0] = 0

    # find maximum gust
    maxid = gust.argmax(axis=1)
    I = np.indices(maxid.shape)
    gust_max = gust[I,maxid].squeeze()


    if mode in ['lb']:
    #if mode in ['lb', 'lb_rho']:
        maxid = gust_lb.argmax(axis=1)
        I = np.indices(maxid.shape)
        gust_max_orig = gust_lb[I,maxid].squeeze()
        suptitle = 'PHY BRALB  '
        plot_name_title = 'applied_phys_bralb_'
    else:
        maxid = gust_est.argmax(axis=1)
        I = np.indices(maxid.shape)
        gust_max_orig = gust_est[I,maxid].squeeze()
        suptitle = 'PHY BRAES  '
        plot_name_title = 'applied_phys_braes_'

    plot_error(obs_gust, model_mean_hr, obs_mean, gust_max, gust_max_orig)
    plt.suptitle(suptitle + mode)

    if i_plot == 1:
        plt.show()
    elif i_plot > 1:
        if i_label == '':
            plot_name = CNapply.plot_path + plot_name_title +str(mode)+'.png'
        else:
            plot_name = CNapply.plot_path + plot_name_title+str(i_label)+'_'+str(mode)+'.png'
        print(plot_name)
        plt.savefig(plot_name)
        plt.close('all')


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from plot_functions import plot_error, plot_mod_vs_obs
import globals as G
from namelist_cases import Case_Namelist
import namelist_cases as nl
from datetime import timedelta

############ USER INPUT #############
case_index = nl.case_index
CN = Case_Namelist(case_index)
# do not plot (0) show plot (1) save plot (2)
i_plot = nl.i_plot
i_plot_type = nl.i_plot_type
model_dt = nl.model_dt
nhrs_forecast = nl.nhrs_forecast
#i_load = nl.i_load
#i_train = nl.i_train
i_output_error = 1
default_learning_rate_factor = 1E-2
delete_existing_param_file = nl.delete_existing_param_file
#max_mean_wind_error = nl.max_mean_wind_error
#sample_weight = nl.sample_weight

modes = ['ln',
         'nl']

i_mode_ints = range(0,len(modes))
i_mode_ints = [0]
sgd_prob = 1.0
#####################################

###############################################################################
###### PART 0: General Setup
###############################################################################

def box_cox(data, l1, l2):
    if l1 != 0:
        data = (data + l2)**l1 - 1
    else:
        data = np.log(data + l2)
    return(data)

# create directories
if i_plot > 1 and not os.path.exists(CN.plot_path):
    os.mkdir(CN.plot_path)

if delete_existing_param_file:
    try:
        os.remove(CN.params_readj_path)
    except:
        pass

# load data
data = pickle.load( open(CN.mod_path, 'rb') )

obs_mean = data['obs_mean']
obs_gust = data['obs_gust']
zvp10   = data['model_fields']['zvp10']
tcm     = data['model_fields']['tcm']


###############################################################################
###### PART 1: Preprocessing
###############################################################################

# Process fields
tcm[tcm < 5E-4] = 5E-4
tcm = np.sqrt(tcm)

## observation to 1D and filter values
model_mean = np.mean(zvp10, axis=2)

# get obs nan mask
obsmask = np.isnan(obs_gust)
obsmask[np.isnan(obs_mean)] = True

# mask remaining arrays
obs_gust = obs_gust[~obsmask] 
obs_mean = obs_mean[~obsmask]
model_mean = model_mean[~obsmask]
tcm = tcm[~obsmask]
zvp10 = zvp10[~obsmask]


# transformations
## arcsin
#obs_gust = obs_gust/np.max(obs_gust)
#obs_gust = np.sqrt(obs_gust)
#obs_gust = np.arcsin(obs_gust)

# log
#obs_gust[obs_gust == 0] = 0.1
#obs_gust = np.log(obs_gust)

## box-cox lambda1=0, lambda2=1
#l1 = 0.25
#l2 = 1
#obs_gust = box_cox(obs_gust, l1, l2)

# sklearn box-cox
#from sklearn.preprocessing import power_transform
#obs_gust[obs_gust < 0.1] = 0.1
#obs_gust = obs_gust.reshape(-1,1)
#obs_gust = power_transform(obs_gust, method='box-cox') 
#obs_gust = power_transform(obs_gust, method='yeo-johnson') 

#from scipy import stats

#zvp10tcm = zvp10*tcm
#shift = 0.1
#tmp,lamb = stats.boxcox(zvp10tcm.flatten()+shift)
#zvp10tcm = box_cox(zvp10tcm, lamb, shift)

#shift = 0.1
#tmp,lamb = stats.boxcox(zvp10.flatten()+shift)
#zvp10 = box_cox(zvp10, lamb, shift)

#shift = 0.1
#tmp,lambda_gust = stats.boxcox(obs_gust+shift)
#obs_gust = box_cox(obs_gust, lambda_gust, shift)
#plt.hist(obs_gust, bins=30)
#plt.show()
#quit()

#shift = 0.1
#tmp,lamb = stats.boxcox(zvp10.flatten()+shift)
#zvp10 = box_cox(zvp10, lamb, shift)

#plt.hist(zvp10.flatten(), bins=30)
#plt.show()
#quit()

#quit()

###############################################################################
###### PART 2: Training
###############################################################################


def find_hourly_max(gust, fields=None):
    if fields is not None: 
        # find maximum gust
        maxid = gust.argmax(axis=1)
        I = np.indices(maxid.shape)
        gust_max = gust[I,maxid].squeeze()
        max_fields = {}
        for field_name,value in fields.items():
            max_fields[field_name] = fields[field_name][I,maxid].squeeze()
        return(gust_max, max_fields)
    else:
        gust_max = np.max(gust, axis=1)
        return(gust_max)
    


from tuning_functions import rotate
def plot_rotation(mod, obs):
    plt.scatter(mod, obs, marker='.', color='black')
    plt.xlim((0,60))
    plt.ylim((-20,60))
    plt.axhline(y=0, color='grey')
    plt.xlabel('model')
    plt.ylabel('obs')


    xy = np.row_stack( [mod, obs] )
    xy_rot = rotate(xy, -np.pi/4)
    #quit()
    speed = xy_rot[0,:]
    error = xy_rot[1,:]

    plt.scatter(speed, error, marker='.', color='orange')
    #plt.show()
    #quit()

    return(speed, error)


fields = {}
fields['zvp10'] = zvp10
fields['zvp10_tcm'] = zvp10*tcm

gust_orig = zvp10 + 7.2*tcm*zvp10
gust_max_orig = find_hourly_max(gust_orig)

# TODO make vectors
alpha = 5
learning_rate = 0.20
n_bins = 8
# BIN NAMELIST
if n_bins == 8:
    bins = [(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80)]
    bin_weights = [1,1,1,3,5,1,1,1]
elif n_bins == 1:
    bins = [(0,80)]
    bin_weights = [1]
elif n_bins == 2:
    bins = [(0,40),(40,80)]
    bin_weights = [1,1]
elif n_bins == 4:
    bins = [(0,20),(20,40),(40,60),(60,80)]
    bin_weights = [1,1,1,1]
n_bins = len(bins)

sin_th = np.sin(-np.pi/4)
cos_th = np.cos(-np.pi/4)

N = len(obs_gust)

for i in range(0,30):
    gust = fields['zvp10'] + alpha*fields['zvp10_tcm']
    gust_max, fields_max = find_hourly_max(gust, fields)

    #speed, error = plot_rotation(gust_max, obs_gust)
    #plt.show()
    #quit()

    #x = np.asarray([0,1,2,3])
    #y = np.asarray([0,1,2,3])
    #y = np.asarray([0,2,4,6])
    #xy = np.row_stack( [x, y] )
    #xy_rot = rotate(xy, -np.pi/4)
    #plot_rotation(x, y)
    #plt.show()
    #quit()


    # TODO what needs to be done here to get in agreement with theory?
    speed       =   - sin_th * gust_max + cos_th * obs_gust
    model_error = -(  sin_th * gust_max + cos_th * obs_gust )

    #error_test = sin_th * fields_max['zvp10'] + \
    #             alpha * sin_th * fields_max['zvp10_tcm'] + \
    #             cos_th * obs_gust
    #print(np.mean(error_now))
    #print(np.mean(error_test))
    #quit()
    #plt.scatter(speed_now, error_now, color='red', marker='.')
    #plt.show()

    tot_weight = 0
    dalpha = 0
    for bI in range(0,n_bins):
        #bI = 4

        bin_inds = np.argwhere((speed >= bins[bI][0]) & \
                               (speed < bins[bI][1])).squeeze()

        model_error_bin = model_error[bin_inds]
        #bin_obs_gust = obs_gust.squeeze()[bin_inds]
        fields_bin = {}
        fields_bin['zvp10_tcm'] = fields_max['zvp10_tcm'][bin_inds]
        try:
            N_bin = len(model_error_bin)
        except TypeError: # in case model_error_bin is scalar not array
            N_bin = 1

        if N_bin > 20:
            #print('bin number: \t' + str(bI))
            #print('n samples: \t' + str(N_bin))

            # calculate derivative of loss function with respect to alpha
            #dalpha = 2/N * np.sum(model_error * sin_th * fields_max['zvp10_tcm'])
            dalpha += bin_weights[bI] * \
                      2/N_bin * np.sum(model_error_bin * sin_th *
                                        fields_bin['zvp10_tcm'])
            tot_weight += bin_weights[bI]
    dalpha = dalpha/tot_weight

    # update alpha
    alpha += learning_rate*dalpha
    print(alpha)


















gust = fields['zvp10'] + alpha*fields['zvp10_tcm']
gust_max = find_hourly_max(gust)

#quit()












#fields = {}
#fields['zvp10'] = zvp10
#fields['zvp10_tcm'] = zvp10*tcm
#fields['zvp10_2_tcm'] = (zvp10**2)*tcm
#
#
#gust_orig = zvp10 + 7.2*tcm*zvp10 + 0.09*tcm*zvp10**2
#gust_max_orig = find_hourly_max(gust_orig)
#
#
#alpha1 = 2
#alpha2 = 0
#learning_rate_1 = 0.5
#learning_rate_2 = 0.01
#
#for i in range(0,20):
#    gust = fields['zvp10'] + alpha1*fields['zvp10_tcm'] + \
#            alpha2*fields['zvp10_2_tcm']
#    gust_max, fields_max = find_hourly_max(gust, fields)
#
#    #speed, error = plot_rotation(gust_max, obs_gust)
#
#    #x = np.asarray([0,1,2,3])
#    #y = np.asarray([0,1,2,3])
#    #xy = np.row_stack( [x, y] )
#    #xy_rot = rotate(xy, -np.pi/4)
#    #plot_rotation(x, y)
#
#    sin_th = np.sin(-np.pi/4)
#    cos_th = np.cos(-np.pi/4)
#
#    N = len(obs_gust)
#
#
#    # TODO what needs to be done here to get in agreement with theory?
#    speed_now = - sin_th * gust_max + cos_th * obs_gust
#    error_now = sin_th * gust_max + cos_th * obs_gust
#
#    #error_test = sin_th * fields_max['zvp10'] + \
#    #             alpha * sin_th * fields_max['zvp10tcm'] + \
#    #             cos_th * obs_gust
#    #print(np.mean(error_now))
#    #print(np.mean(error_test))
#    #quit()
#    #plt.scatter(speed_now, error_now, color='red', marker='.')
#    #plt.show()
#
#    model_error = - error_now
#
#    dalpha1 = 2/N * np.sum(model_error * sin_th * fields_max['zvp10_tcm'])
#    dalpha2 = 2/N * np.sum(model_error * sin_th * fields_max['zvp10_2_tcm'])
#    print('dalpha1 ' + str(dalpha1) + ' dalpha2 ' + str(dalpha2))
#
#    alpha1 += learning_rate_1*dalpha1
#    alpha2 += learning_rate_2*dalpha2
#    print('alpha1 ' + str(alpha1) + ' alpha2 ' + str(alpha2))
#
#
#gust = fields['zvp10'] + alpha1*fields['zvp10_tcm'] + alpha2*fields['zvp10_2_tcm']
#gust_max = find_hourly_max(gust)
#
#
##quit()








# TODO debug
mode = 'test'


# PLOT
if i_plot > 0:
    if i_plot_type == 0:
        plot_error(obs_gust, model_mean, obs_mean, gust_max, gust_max_orig)
    elif i_plot_type == 1:
        plot_mod_vs_obs(obs_gust, gust_max, gust_max_orig, obs_mean, model_mean)
    else:
        raise NotImplementedError()
    plt.suptitle('READJUST  '+mode)

    if i_plot == 1:
        plt.show()
    elif i_plot > 1:
        if i_plot_type == 0:
            plot_name = CN.plot_path + 'tuning_readj_'+str(mode)+'.png'
        elif i_plot_type == 1:
            plot_name = CN.plot_path + 'plot1_tuning_readj_'+str(mode)+'.png'
        print(plot_name)
        plt.savefig(plot_name)
        plt.close('all')




quit()



for mode_int in i_mode_ints:
    mode = modes[mode_int]
    print('#################################################################################')
    print('############################## ' + str(mode) + ' ################################')

    # initial gust
    if mode == 'ln':
        gust = zvp10 + 7.2*tcm*zvp10
    elif mode == 'nl':
        gust = zvp10 + 7.2*tcm*zvp10 + 0.09*tcm*zvp10**2
    gust_max_orig = np.amax(gust,axis=1)

    if mode == 'nl':
        learning_rate_factor = default_learning_rate_factor * 1/20
        d_error_thresh = 1E-6
    else:
        learning_rate_factor = default_learning_rate_factor
        d_error_thresh = 1E-5

    alpha1 = 7
    alpha2 = 0
    error_old = np.Inf
    d_errors = np.full(int(1/sgd_prob*5), 100.)
    learning_rate = 1E-5

    c = 0
    while np.abs(np.mean(d_errors)) > d_error_thresh:


        # SGD selection
        sgd_inds = np.random.choice([True, False], (zvp10.shape[0]), p=[sgd_prob,1-sgd_prob])
        #sgd_zvp10tcm = zvp10[sgd_inds,:]*tcm[sgd_inds,:]
        sgd_zvp10 = zvp10[sgd_inds,:]
        sgd_tcm = tcm[sgd_inds,:]
        sgd_obs_gust = obs_gust[sgd_inds]
        N = len(sgd_obs_gust)

        # calc current time step gusts
        if mode == 'ln':
            sgd_gust = sgd_zvp10 + alpha1*sgd_tcm*sgd_zvp10
            #sgd_gust = sgd_zvp10 + alpha1*sgd_zvp10tcm
        elif mode == 'nl':
            sgd_gust = sgd_zvp10 + alpha1*sgd_tcm*sgd_zvp10 + alpha2*sgd_tcm*sgd_zvp10**2
        else:
            raise ValueError('wrong mode')

        # find maximum gust
        maxid = sgd_gust.argmax(axis=1)
        I = np.indices(maxid.shape)
        sgd_tcm_max = sgd_tcm[I,maxid]
        sgd_zvp10_max = sgd_zvp10[I,maxid]
        #sgd_zvp10tcm_max = sgd_zvp10tcm[I,maxid]
        sgd_gust_max = sgd_gust[I,maxid]


        # error
        deviation = sgd_obs_gust - sgd_gust_max
        error_now = np.sqrt(np.sum(deviation**2)/N)
        d_error = error_old - error_now
        d_errors = np.roll(d_errors, shift=1)
        d_errors[0] = d_error
        error_old = error_now
        if i_output_error:
            if c % 10 == 0:
                print(str(c) + '\t' + str(error_now) + '\t' + str(np.abs(np.mean(d_errors))))
                print('alpha 1 ' + str(alpha1) + ' alpha 2 ' + str(alpha2))

        # gradient of parameters
        if mode == 'ln':
            dalpha1 = -2/N * np.sum( sgd_tcm_max*sgd_zvp10_max * deviation )
            #dalpha1 = -2/N * np.sum( sgd_zvp10tcm_max * deviation )
            dalpha2 = 0
        elif mode == 'nl':
            dalpha1 = -2/N * np.sum( sgd_tcm_max*sgd_zvp10_max    * deviation )
            dalpha2 = -2/N * np.sum( sgd_tcm_max*sgd_zvp10_max**2 * deviation )
        else:
            raise ValueError('wrong mode')

        alpha1 = alpha1 - learning_rate * dalpha1
        alpha2 = alpha2 - learning_rate * dalpha2

        # adjust learning rate
        learning_rate = error_now*learning_rate_factor

        c += 1


    alpha1 = 10

    print('############')
    print('alpha1 ' + str(alpha1))
    print('alpha2 ' + str(alpha2))
    print('############')



    # Calculate final gust
    if mode == 'ln':
        gust = zvp10 + alpha1*tcm*zvp10
        #gust = zvp10 + alpha1*zvp10tcm
    elif mode == 'nl':
        gust = zvp10 + alpha1*tcm*zvp10 + alpha2*tcm*zvp10**2
    else:
        raise ValueError('wrong mode')
    # find maximum gust
    maxid = gust.argmax(axis=1)
    I = np.indices(maxid.shape)
    gust_max = gust[I,maxid].squeeze()


    # PLOT
    if i_plot > 0:
        if i_plot_type == 0:
            plot_error(obs_gust, model_mean, obs_mean, gust_max, gust_max_orig)
        elif i_plot_type == 1:
            plot_mod_vs_obs(obs_gust, gust_max, gust_max_orig, obs_mean, model_mean)
        else:
            raise NotImplementedError()
        plt.suptitle('READJUST  '+mode)

        if i_plot == 1:
            plt.show()
        elif i_plot > 1:
            if i_plot_type == 0:
                plot_name = CN.plot_path + 'tuning_readj_'+str(mode)+'.png'
            elif i_plot_type == 1:
                plot_name = CN.plot_path + 'plot1_tuning_readj_'+str(mode)+'.png'
            print(plot_name)
            plt.savefig(plot_name)
            plt.close('all')


    # RESCALE ALPHA VALUES
    # not scaled

    # SAVE PARAMETERS 
    if os.path.exists(CN.params_readj_path):
        params = pickle.load( open(CN.params_readj_path, 'rb') )
    else:
        params = {}
    params[mode] = {'alphas':{'1':alpha1,'2':alpha2}}
    pickle.dump(params, open(CN.params_readj_path, 'wb'))



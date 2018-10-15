############################
# TRAIN SCRIPT SETTINGS
# 10 all
# 11 June_18
# 12 December_17
# 13 June 18 and December 17
case_index      = 4
# do not plot (0) show plot (1) save plot (2)
i_plot          = 2
i_plot_type     = 1
model_dt        = 10
nhrs_forecast   = 24
i_load          = 1
i_train         = 1
delete_existing_param_file = 1
#max_mean_wind_error = 10000.0
#sample_weight = 'linear'
#sample_weight = 'squared'
#sample_weight = '1'
############################
# APPLY SCRIPT SETTINGS
train_case_index    = 10
apply_case_index    = 13
apply_i_plot        = 2
apply_model_dt      = 10
############################
class Case_Namelist:

    exp_id = 101
    #exp_id = 105


    # path to raw observation data (as obtained by syn_get_obs script)
    raw_obs_folder = '../obs_out/'
    # path to raw model output
    raw_mod_folder = '../model_out/'
    # folder where all the data except raw obs files is saved
    data_folder = '../data/'
    # folder where the trained parameters are stored
    param_folder = data_folder + 'params/'
    # plots are put here
    plot_base_dir = '../plots/'

   
    # 01_prep_obs: path to file containing meta information about stations (mainly column 'Use' is interesting)
    #stations_meta_path = '../obs_out/ps_fkl010b1_2262.csv'
    stations_meta_path = '../obs_out/ps_fkl010b1_2262_SMN.csv'


    OBS_RAW = 'obs_raw'
    OBS = 'obs_case'
    MOD_RAW = 'mod_raw'
    MOD = 'mod_run'


    cases = [
        # 0 DEBUG
        {OBS_RAW:['20180102sfc.','20180103sfc.','20180104sfc.'],
                    OBS:'20180103_Burglind', MOD_RAW:'DEBUG_Burglind'},

        ####################################################################################
        ############################# TRAINING CASES
        ####################################################################################
        # 1 Zeus 17
        {OBS_RAW:['20170304sfc.','20170305sfc.','20170306sfc.','20170307sfc.','20170308sfc.'],
                    OBS:'20170305_Zeus', MOD_RAW:'20170305_Zeus'},
        # 2 Konvektion 17
        {OBS_RAW:['20170719sfc.'],
                    OBS:'20170719_Konvektion', MOD_RAW:'20170719_Konvektion'},
        # 3 Sommersturm 17
        {OBS_RAW:['20170723sfc.','20170724sfc.','20170725sfc.','20170726sfc.'],
                    OBS:'20170724_Sommersturm', MOD_RAW:'20170724_Sommersturm'},
        # 4 Burglind
        {OBS_RAW:['20180102sfc.','20180103sfc.','20180104sfc.'],
                    OBS:'20180103_Burglind', MOD_RAW:'20180103_Burglind'},
        # 5 Friederike
        {OBS_RAW:['20180115sfc.','20180116sfc.','20180117sfc.','20180118sfc.','20180119sfc.'],
                    OBS:'20180116_Friederike', MOD_RAW:'20180116_Friederike'},
        # 6 Foehntaeler
        {OBS_RAW:['20180228sfc.','20180301sfc.','20180302sfc.'],
                    OBS:'20180301_Foehntaeler', MOD_RAW:'20180301_Foehntaeler'},
        # 7 Foehnsturm
        {OBS_RAW:['20180428sfc.','20180429sfc.','20180430sfc.','20180501sfc.'],
                    OBS:'20180429_Foehnsturm', MOD_RAW:'20180429_Foehnsturm'},
        # 8 Bisensturm
        {OBS_RAW:['20180502sfc.','20180503sfc.','20180504sfc.','20180505sfc.'],
                    OBS:'20180503_Bisensturm', MOD_RAW:'20180503_Bisensturm'},
        # 9 Gewittertage
        {OBS_RAW:['20180529sfc.','20180530sfc.','20180531sfc.','20180601sfc.'],
                    OBS:'20180530_Gewittertage', MOD_RAW:'20180530_Gewittertage'},
        # 10 all
        {OBS_RAW:['20170304sfc.','20170305sfc.','20170306sfc.','20170307sfc.','20170308sfc.',
                  '20170719sfc.',
                  '20170723sfc.','20170724sfc.','20170725sfc.','20170726sfc.',
                  '20180102sfc.','20180103sfc.','20180104sfc.',
                  '20180115sfc.','20180116sfc.','20180117sfc.','20180118sfc.','20180119sfc.',
                  '20180228sfc.','20180301sfc.','20180302sfc.',
                  '20180428sfc.','20180429sfc.','20180430sfc.','20180501sfc.',
                  '20180502sfc.','20180503sfc.','20180504sfc.','20180505sfc.',
                  '20180529sfc.','20180530sfc.','20180531sfc.','20180601sfc.'],
                    OBS:'All', MOD_RAW:'All'},
        # 11 June_18
        {OBS_RAW:['20180620sfc.','20180621sfc.','20180622sfc.','20180623sfc.',
                  '20180624sfc.','20180625sfc.','20180626sfc.','20180627sfc.',
                  '20180628sfc.','20180629sfc.','20180630sfc.','20180701sfc.'],
                    OBS:'June_18', MOD_RAW:'June_18'},
        # 12 December_17
        {OBS_RAW:['20171206sfc.','20171207sfc.','20171208sfc.','20171209sfc.',
                  '20171210sfc.','20171211sfc.','20171212sfc.','20171213sfc.',
                  '20171214sfc.','20171215sfc.','20171216sfc.','20171217sfc.'],
                    OBS:'December_17', MOD_RAW:'December_17'},
        # 13 June 18 and December 17
        {OBS_RAW:['20180620sfc.','20180621sfc.','20180622sfc.','20180623sfc.',
                  '20180624sfc.','20180625sfc.','20180626sfc.','20180627sfc.',
                  '20180628sfc.','20180629sfc.','20180630sfc.','20180701sfc.',
                  '20171206sfc.','20171207sfc.','20171208sfc.','20171209sfc.',
                  '20171210sfc.','20171211sfc.','20171212sfc.','20171213sfc.',
                  '20171214sfc.','20171215sfc.','20171216sfc.','20171217sfc.'],
                    OBS:'Eval_June_Dec', MOD_RAW:'Eval_June_Dec'},


        ####################################################################################
        ############################# TESTING AND EXPERIMENTAL CASES
        ####################################################################################

        # 14 test case for gust model output
        {OBS_RAW:['20180103sfc.'],
                OBS:'test_gust', MOD_RAW:'test_gust'},
        # 15 December_SMN
        {OBS_RAW:['20171206sfc.','20171207sfc.','20171208sfc.','20171209sfc.',
                  '20171210sfc.','20171211sfc.','20171212sfc.','20171213sfc.',
                  '20171214sfc.','20171215sfc.','20171216sfc.','20171217sfc.'],
                    OBS:'December_SMN', MOD_RAW:'December_SMN'},
        # 16 Burglind ANAC1
        {OBS_RAW:['20180102sfc.','20180103sfc.','20180104sfc.'],
                    OBS:'20180103_Burglind_ANAC1', MOD_RAW:'20180103_Burglind_ANAC1'},

        # 17 test case 18010300 for gust testing with Guy's file VBM10M_all
        {OBS_RAW:['20180103sfc.'],
                OBS:'test_gust_burglind', MOD_RAW:'test_gust_burglind'},

        # 18 all_SMN
        {OBS_RAW:['20170304sfc.','20170305sfc.','20170306sfc.','20170307sfc.','20170308sfc.',
                  '20170719sfc.',
                  '20170723sfc.','20170724sfc.','20170725sfc.','20170726sfc.',
                  '20180102sfc.','20180103sfc.','20180104sfc.',
                  '20180115sfc.','20180116sfc.','20180117sfc.','20180118sfc.','20180119sfc.',
                  '20180228sfc.','20180301sfc.','20180302sfc.',
                  '20180428sfc.','20180429sfc.','20180430sfc.','20180501sfc.',
                  '20180502sfc.','20180503sfc.','20180504sfc.','20180505sfc.',
                  '20180529sfc.','20180530sfc.','20180531sfc.','20180601sfc.'],
                    OBS:'All_SMN', MOD_RAW:'All_SMN'},

    ]

    def __init__(self, case_ind):

        self.raw_obs_path = []
        for element in self.cases[case_ind][self.OBS_RAW]:
            self.raw_obs_path.append(self.raw_obs_folder + element)

        self.obs_path = self.data_folder + 'OBS_' + self.cases[case_ind][self.OBS] + '.pkl'

        self.raw_mod_path = self.raw_mod_folder + self.cases[case_ind][self.MOD_RAW] + '/' + str(self.exp_id) + '/'

        self.mod_path = self.data_folder + 'OBS_' + self.cases[case_ind][self.OBS] + '_EXPID_' + \
                        str(self.exp_id) + '.pkl'

        self.plot_path = self.plot_base_dir + 'OBS_' + self.cases[case_ind][self.OBS] + '_EXPID_' + \
                        str(self.exp_id) + '/'

        self.case_name = 'OBS_' + self.cases[case_ind][self.OBS] + '_EXPID_' + \
                        str(self.exp_id) 

        #### TRAIN

        self.train_readj_path = self.data_folder + 'train_readj_OBS_' + self.cases[case_ind][self.OBS] + '_EXPID_' + \
                        str(self.exp_id) + '.pkl'

        self.train_stat_path = self.data_folder + 'train_stat_OBS_' + self.cases[case_ind][self.OBS] + '_EXPID_' + \
                        str(self.exp_id) + '.pkl'

        self.train_braes_path = self.data_folder + 'train_braes_OBS_' + self.cases[case_ind][self.OBS] + '_EXPID_' + \
                        str(self.exp_id) + '.pkl'

        self.train_bralb_path = self.data_folder + 'train_bralb_OBS_' + self.cases[case_ind][self.OBS] + '_EXPID_' + \
                        str(self.exp_id) + '.pkl'

        self.train_icon_path = self.data_folder + 'train_icon_OBS_' + self.cases[case_ind][self.OBS] + '_EXPID_' + \
                        str(self.exp_id) + '.pkl'

        self.phys_bra_path = self.data_folder + 'train_phys_OBS_' + self.cases[case_ind][self.OBS] + '_EXPID_' + \
                        str(self.exp_id) + '.pkl'

        self.train_braub_path = self.data_folder + 'train_braub_OBS_' + self.cases[case_ind][self.OBS] + '_EXPID_' + \
                        str(self.exp_id) + '.pkl'

        # new readjustment approach
        self.train_readjNEW_path = self.data_folder + 'train_readjNEW_OBS_' + self.cases[case_ind][self.OBS] + '_EXPID_' + \
                        str(self.exp_id) + '.pkl'

        #### ML

        self.ML_braes_path = self.data_folder + 'ML_braes_OBS_' + self.cases[case_ind][self.OBS] + '_EXPID_' + \
                        str(self.exp_id) + '.pkl'


        #### PARAMETERS

        self.params_readj_path = self.param_folder + 'readj_OBS_' + self.cases[case_ind][self.OBS] + '_EXPID_' + \
                        str(self.exp_id) + '.pkl'

        self.params_stat_path = self.param_folder + 'stat_OBS_' + self.cases[case_ind][self.OBS] + '_EXPID_' + \
                        str(self.exp_id) + '.pkl'

        self.params_braes_path = self.param_folder + 'braes_OBS_' + self.cases[case_ind][self.OBS] + '_EXPID_' + \
                        str(self.exp_id) + '.pkl'

        self.params_bralb_path = self.param_folder + 'bralb_OBS_' + self.cases[case_ind][self.OBS] + '_EXPID_' + \
                        str(self.exp_id) + '.pkl'

        self.params_icon_path = self.param_folder + 'icon_OBS_' + self.cases[case_ind][self.OBS] + '_EXPID_' + \
                        str(self.exp_id) + '.pkl'

        self.params_phys_bra_path = self.param_folder + 'phys_OBS_' + self.cases[case_ind][self.OBS] + '_EXPID_' + \
                        str(self.exp_id) + '.pkl'

        self.params_braub_path = self.param_folder + 'braub_OBS_' + self.cases[case_ind][self.OBS] + '_EXPID_' + \
                        str(self.exp_id) + '.pkl'


        print('##########################################')
        print(self.case_name)
        print('##########################################')



class Case_Namelist:

    # path to raw observation data (as obtained by syn_get_obs script)
    raw_obs_folder = '../obs_out/'
    # path to raw model output
    raw_mod_folder = '../model_out/'
    # folder where all the data except raw obs files is saved
    data_folder = '../data/'
    # plots are put here
    plot_base_dir = '../plots/'

   
    # 01_prep_obs: path to file containing meta information about stations (mainly column 'Use' is interesting)
    stations_meta_path = '../obs_out/ps_fkl010b1_2262.csv'


    OBS_RAW = 'obs_raw'
    OBS = 'obs_case'
    MOD_RAW = 'mod_raw'
    MOD = 'mod_run'


    cases = [
        ## 0 DEBUG
        #{OBS_RAW:['20180610sfc.'],
        #            OBS:'20180103_Sommer', MOD_RAW:'DEBUG_Sommer', MOD:'DEBUG'},
        # 0 DEBUG
        {OBS_RAW:['20180102sfc.','20180103sfc.','20180104sfc.'],
                    OBS:'20180103_Burglind', MOD_RAW:'DEBUG_Burglind', MOD:'DEBUG'},
        # 1 Burglind
        {OBS_RAW:['20180102sfc.','20180103sfc.','20180104sfc.'],
                    OBS:'20180103_Burglind', MOD_RAW:'20180103_Burglind', MOD:'ref'},
        # 2 Bisensturm
        {OBS_RAW:['20180502sfc.','20180503sfc.','20180504sfc.','20180505sfc.'],
                    OBS:'20180503_Bisensturm', MOD_RAW:'20180503_Bisensturm', MOD:'ref'},
        # 3 Foehnsturm
        {OBS_RAW:['20180428sfc.','20180429sfc.','20180430sfc.','20180501sfc.'],
                    OBS:'20180429_Foehnsturm', MOD_RAW:'20180429_Foehnsturm', MOD:'ref'},
        # 4 Friederike
        {OBS_RAW:['20180115sfc.','20180116sfc.','20180117sfc.','20180118sfc.','20180119sfc.'],
                    OBS:'20180116_Friederike', MOD_RAW:'20180116_Friederike', MOD:'ref'},
        # 5 Zeus 17
        {OBS_RAW:['20170304sfc.','20170305sfc.','20170306sfc.','20170307sfc.','20170308sfc.'],
                    OBS:'20170305_Zeus', MOD_RAW:'20170305_Zeus', MOD:'ref'},
        # 6 Sommersturm 17
        {OBS_RAW:['20170723sfc.','20170724sfc.','20170725sfc.','20170726sfc.'],
                    OBS:'20170724_Sommersturm', MOD_RAW:'20170724_Sommersturm', MOD:'ref'},
        # 7 JanuaryDays
        {OBS_RAW:['20180119sfc.','20180120sfc.','20180121sfc.','20180122sfc.','20180123sfc.',
                    '20180124sfc.','20180125sfc.','20180126sfc.','20180127sfc.','20180128sfc.'],
                    OBS:'20180119_JanuaryDays', MOD_RAW:'20180119_JanuaryDays', MOD:'ref'},
        # 8 all
        {OBS_RAW:['20180102sfc.','20180103sfc.','20180104sfc.',
                  '20180502sfc.','20180503sfc.','20180504sfc.','20180505sfc.',
                  '20180428sfc.','20180429sfc.','20180430sfc.','20180501sfc.',
                  '20180115sfc.','20180116sfc.','20180117sfc.','20180118sfc.','20180119sfc.',
                  '20170304sfc.','20170305sfc.','20170306sfc.','20170307sfc.','20170308sfc.',
                  '20170723sfc.','20170724sfc.','20170725sfc.','20170726sfc.'],
                    OBS:'All', MOD_RAW:'All', MOD:'ref'}
    ]

    def __init__(self, case_ind):

        self.raw_obs_path = []
        for element in self.cases[case_ind][self.OBS_RAW]:
            self.raw_obs_path.append(self.raw_obs_folder + element)

        self.obs_path = self.data_folder + 'OBS_' + self.cases[case_ind][self.OBS] + '.pkl'

        self.raw_mod_path = self.raw_mod_folder + self.cases[case_ind][self.MOD_RAW] + '/'

        self.mod_path = self.data_folder + 'OBS_' + self.cases[case_ind][self.OBS] + '_MOD_' + \
                        self.cases[case_ind][self.MOD] + '.pkl'

        self.plot_path = self.plot_base_dir + 'OBS_' + self.cases[case_ind][self.OBS] + '_MOD_' + \
                        self.cases[case_ind][self.MOD] + '/'

        self.case_name = 'OBS_' + self.cases[case_ind][self.OBS] + '_MOD_' + \
                        self.cases[case_ind][self.MOD]

        #### TRAIN

        self.train_readj_path = self.data_folder + 'train_readj_OBS_' + self.cases[case_ind][self.OBS] + '_MOD_' + \
                        self.cases[case_ind][self.MOD] + '.pkl'

        self.train_stat_path = self.data_folder + 'train_stat_OBS_' + self.cases[case_ind][self.OBS] + '_MOD_' + \
                        self.cases[case_ind][self.MOD] + '.pkl'

        self.train_bra_path = self.data_folder + 'train_bra_OBS_' + self.cases[case_ind][self.OBS] + '_MOD_' + \
                        self.cases[case_ind][self.MOD] + '.pkl'

        self.train_bralb_path = self.data_folder + 'train_bralb_OBS_' + self.cases[case_ind][self.OBS] + '_MOD_' + \
                        self.cases[case_ind][self.MOD] + '.pkl'

        self.train_icon_path = self.data_folder + 'train_icon_OBS_' + self.cases[case_ind][self.OBS] + '_MOD_' + \
                        self.cases[case_ind][self.MOD] + '.pkl'

        self.phys_bralb_path = self.data_folder + 'phys_bralb_OBS_' + self.cases[case_ind][self.OBS] + '_MOD_' + \
                        self.cases[case_ind][self.MOD] + '.pkl'

        print('##########################################')
        print(self.case_name)
        print('##########################################')


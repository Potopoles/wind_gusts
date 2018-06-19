
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
        # 4 all
        {OBS_RAW:['20180102sfc.','20180103sfc.','20180104sfc.',
                  '20180502sfc.','20180503sfc.','20180504sfc.','20180505sfc.',
                  '20180428sfc.','20180429sfc.','20180430sfc.','20180501sfc.'],
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

        print('##########################################')
        print(self.case_name)
        print('##########################################')


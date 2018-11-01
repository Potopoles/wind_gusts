import numpy as np
import os

# bad models
#['zvp10fix', 'zvp10_tcm', 'zvp10_2_tcm'],
#['zvp10', 'zvp10_z0_2', 'IFS'],
#['zvp10fix', 'IFS'],
#['zvp10', 'IFS'],
#['zvp10', 'zvp10_wdir', 'zvp10_z0', 'IFS'],
#['zvp10', 'zvp10_tcm', 'zvp10_2_tcm'],
#['zvp10', 'zvp10_tcm', 'braes'],
#['zvp10', 'zvp10_tcm', 'braes', 'braes_2'],
#['zvp10', 'zvp10_tcm', 'braes', 'braes_2', 'braes_3'],
#['zvp10', 'zvp10_tcm', 'bralb'],
#['zvp10', 'zvp10_tcm', 'bralb', 'bralb_2'],
#['zvp10', 'zvp10_tcm', 'bralb', 'bralb_2', 'bralb_3'],

# bad features ['zvp10_IFS']
# bad features ['tke']

class Linear_Models:

    def __init__(self, PR, CN, reset_model_constellation, use,
                predictor_list=None):

        #use = [0]
        #use = np.arange(2,4)

        good_models = [
            ['zvp10fix', 'zvp10_tcm'],
            ['zvp10', 'zvp10_tcm'],

            ['zvp10', 'zvp10_tcm',
                'zvp10_braes', 'zvp10_braes_2', 'zvp10_braes_3'],
            ['zvp10', 'zvp10_tcm',
                'zvp10_bralb', 'zvp10_bralb_2', 'zvp10_bralb_3'],
            ['zvp10', 'zvp10_tcm',
                'zvp10_braub', 'zvp10_braub_2', 'zvp10_braub_3'],

            ['zvp10', 'zvp10_tcm',
                'zvp10_zbraes', 'zvp10_zbraes_2', 'zvp10_zbraes_3'],
            ['zvp10', 'zvp10_tcm',
                'zvp10_zbralb', 'zvp10_zbralb_2', 'zvp10_zbralb_3'],

            ['zvp10', 'zvp10_tcm',
                'zvp10_braes', 'zvp10_braes_2', 'zvp10_braes_3',
                'zvp10_zbraes', 'zvp10_zbraes_2'],
            ['zvp10', 'zvp10_tcm',
                'zvp10_bralb', 'zvp10_bralb_2', 'zvp10_bralb_3',
                'zvp10_zbralb', 'zvp10_zbralb_2'],

            ['zvp10', 'zvp10_tcm',
                'zvp10_bralb', 'zvp10_bralb_2', 'zvp10_bralb_3',
                'zvp10_zbralb', 'zvp10_zbralb_2',
                'zvp10_zbraes', 'zvp10_zbraub'],

            ['zvp10', 'zvp10_tcm', 'zvp10_z0',  
               'zvp10_bralb', 'zvp10_bralb_2', 'zvp10_bralb_3',
               'zvp10_zbralb', 'zvp10_zbralb_2',
               'zvp10_zbraes', 'zvp10_zbraub'],

            ['zvp10', 'zvp10_tcm',
                'zvp10_bralb', 'zvp10_bralb_2', 'zvp10_bralb_3',
                'zvp10_zbralb', 'zvp10_zbralb_2', 'zvp10_zbraes'],

            ['zvp10', 'zvp10_tcm',
                'zvp10_hsurf', 'zvp10_hsurf_2', 'zvp10_hsurf_3',
                'zvp10_bralb', 'zvp10_bralb_2', 'zvp10_bralb_3',
                'zvp10_zbralb', 'zvp10_zbralb_2',
                'zvp10_zbraes', 'zvp10_zbraub'],

        ]

        if predictor_list is None:
            predictor_list = [

            ['zvp10', 'zvp10_tcm']
                #'zvp10_zbralb', 'zvp10_zbralb_2', 'zvp10_zbralb_3'],

            #['zvp10fix', 'zvp10_tcm'],
            #['zvp10', 'zvp10_tcm'],

            #['zvp10', 'zvp10_tcm',
            #    'zvp10_braes', 'zvp10_braes_2', 'zvp10_braes_3'],
            #['zvp10', 'zvp10_tcm',
            #    'zvp10_bralb', 'zvp10_bralb_2', 'zvp10_bralb_3'],
            #['zvp10', 'zvp10_tcm',
            #    'zvp10_braub', 'zvp10_braub_2', 'zvp10_braub_3'],

            #['zvp10', 'zvp10_tcm',
            #    'zvp10_zbraes', 'zvp10_zbraes_2', 'zvp10_zbraes_3'],
            #['zvp10', 'zvp10_tcm',
            #    'zvp10_zbralb', 'zvp10_zbralb_2', 'zvp10_zbralb_3'],

            #['zvp10', 'zvp10_tcm',
            #    'zvp10_braes', 'zvp10_braes_2', 'zvp10_braes_3',
            #    'zvp10_zbraes', 'zvp10_zbraes_2'],
            #['zvp10', 'zvp10_tcm',
            #    'zvp10_bralb', 'zvp10_bralb_2', 'zvp10_bralb_3',
            #    'zvp10_zbralb', 'zvp10_zbralb_2'],

            #['zvp10', 'zvp10_tcm',
            #    'zvp10_bralb', 'zvp10_bralb_2', 'zvp10_bralb_3',
            #    'zvp10_zbralb', 'zvp10_zbralb_2',
            #    'zvp10_zbraes', 'zvp10_zbraub'],

            #['zvp10', 'zvp10_tcm', 'zvp10_z0',  
            #   'zvp10_bralb', 'zvp10_bralb_2', 'zvp10_bralb_3',
            #   'zvp10_zbralb', 'zvp10_zbralb_2',
            #   'zvp10_zbraes', 'zvp10_zbraub'],

            #['zvp10', 'zvp10_tcm',
            #    'zvp10_bralb', 'zvp10_bralb_2', 'zvp10_bralb_3',
            #    'zvp10_zbralb', 'zvp10_zbralb_2', 'zvp10_zbraes'],

            #['zvp10', 'zvp10_tcm',
            #    'zvp10_hsurf', 'zvp10_hsurf_2', 'zvp10_hsurf_3',
            #    'zvp10_bralb', 'zvp10_bralb_2', 'zvp10_bralb_3',
            #    'zvp10_zbralb', 'zvp10_zbralb_2',
            #    'zvp10_zbraes', 'zvp10_zbraub'],


                ]

        # in case this is set only update the list of models and exit
        if reset_model_constellation:
            print('Delete plots')
            files = os.listdir(CN.plot_path)
            for file in files:
                if not file == 'grid_point_wind.png':
                    os.remove(CN.plot_path + file)

            print('Clean and update output folder')
            try:
                files = os.listdir(CN.output_path)
            except FileNotFoundError:
                os.mkdir(CN.output_path)
                files = os.listdir(CN.output_path)
            for file in files:
                if not file == 'Session.vim':
                    os.remove(CN.output_path + file)
            file_name = CN.output_path + 'models.info'
            with open(file_name, 'w') as f:
                for i,model in enumerate(predictor_list):
                    model_key = str('{:03d}'.format(i))
                    f.write('{}  {}\n'.format(model_key,model))

            print('Exit')
            quit()


        models = {}
        self.models = models
        for i,pred_names in enumerate(predictor_list):
            if i in use:
                model_key = str('{:03d}'.format(i))
                models[model_key] = self.create_model(pred_names, PR)

        coefs = {}
        self.coefs = coefs

        scores = {}
        self.scores = scores

                




    def create_model(self, pred_names, PR):
        lm_out = {}
        for pred_name in pred_names:
            lm_out[pred_name] = PR.predictor_structure[pred_name]
        return(lm_out)

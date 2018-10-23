
class Linear_Models:

    def __init__(self):

        use = [0,1]

        predictor_list = [
            ['zvp10fix', 'zvp10_tcm', 'tcm'],
            ['zvp10', 'zvp10_tcm'],
            ['zvp10fix', 'zvp10_tcm', 'zvp10_2_tcm'],
            ['zvp10', 'zvp10_tcm', 'zvp10_2_tcm'],
            ['zvp10', 'zvp10_tcm', 'zvp10_braes'],
            ['zvp10', 'zvp10_tcm', 'zvp10_braes',
               'zvp10_braes_2', 'zvp10_braes_3'],
            ['zvp10', 'zvp10_tcm', 'braes', 'braes_2', 'braes_3'],
            ['zvp10', 'bralb', 'bralb_2', 'bralb_3'],
            ['zvp10', 'zvp10_braes'],
            ['zvp10', 'zvp10_tcm', 'bralb', 'braes', 'braub'],
            ['zvp10fix', 'IFS'],
            ['zvp10fix', 'zvp10_tcm', 'IFS']
            ]


        lm = {}
        self.lm = lm

        lm['tcm']               =   {'fix':0,
                                    'prod':[('tcm',1)]
                                    }
        lm['tke']               =   {'fix':0,
                                    'prod':[('tke',1)]
                                    }
        lm['IFS']               =   {'fix':0,
                                    'prod':[('IFS',1)]
                                    }
        #######################################################################
        ###### zvp10
        #######################################################################
        lm['zvp10fix']          =   {'fix':1,
                                    'prod':[('zvp10',1)]
                                    }
        lm['zvp10']             =   {'fix':0,
                                    'prod':[('zvp10',1)]
                                    }
        lm['zvp10_tcm']         =   {'fix':0,
                                    'prod':[('zvp10',1),('tcm',1)]
                                    }
        lm['zvp10_2_tcm']       =   {'fix':0,
                                    'prod':[('zvp10',2),('tcm',1)]
                                    }
        lm['zvp10_bralb']       =   {'fix':0,
                                    'prod':[('zvp10',1),('zv_bra_lb',1)]
                                    }
        lm['zvp10_braes']       =   {'fix':0,
                                    'prod':[('zvp10',1),('zv_bra_es',1)]
                                    }
        lm['zvp10_braub']       =   {'fix':0,
                                    'prod':[('zvp10',1),('zv_bra_ub',1)]
                                    }
        #######################################################################
        ###### bralb
        #######################################################################
        lm['bralb']             =   {'fix':0,
                                    'prod':[('zv_bra_lb',1)]
                                    }
        lm['bralb_2']           =   {'fix':0,
                                    'prod':[('zv_bra_lb',2)]
                                    }
        lm['bralb_3']           =   {'fix':0,
                                    'prod':[('zv_bra_lb',3)]
                                    }
        #######################################################################
        ###### braes
        #######################################################################
        lm['braes']             =   {'fix':0,
                                    'prod':[('zv_bra_es',1)]
                                    }
        lm['zvp10_braes_2']     =   {'fix':0,
                                    'prod':[('zvp10',1),('zv_bra_es',2)]
                                    }
        lm['zvp10_braes_3']     =   {'fix':0,
                                    'prod':[('zvp10',1),('zv_bra_es',3)]
                                    }
        lm['braes_2']           =   {'fix':0,
                                    'prod':[('zv_bra_es',2)]
                                    }
        lm['braes_3']           =   {'fix':0,
                                    'prod':[('zv_bra_es',3)]
                                    }
        #######################################################################
        ###### braub
        #######################################################################
        lm['braub']             =   {'fix':0,
                                    'prod':[('zv_bra_ub',1)]
                                    }


        models = {}
        self.models = models
        for i,pred_names in enumerate(predictor_list):
            if i in use:
                model_key = str('{:03d}'.format(i))
                #models[model_key] = {}
                #models[model_key]['pred_names'] = pred_names
                #models[model_key]['model'] = self.create_model(pred_names)
                models[model_key] = self.create_model(pred_names)

        coefs = {}
        self.coefs = coefs



    def create_model(self, pred_names):
        lm_out = {}
        for pred_name in pred_names:
            lm_out[pred_name] = self.lm[pred_name]
        return(lm_out)

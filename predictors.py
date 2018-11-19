import numpy as np
import globals as G





class Predictors:

    def __init__(self, ncf, data):
        
        # nc file containing all the fields
        self.ncf = ncf
        # pickle binary containing meta data and observations
        self.data = data

        self.array_shape = None

        preproc = {}
        # fields that can be loaded directly without any preprocessing
        # after loading
        no_preproc_fields = ['zvp10', 'zv_bra_lb', 'zv_bra_es', 'zv_bra_ub',
                            'tkel1', 'z0']
        for fld in no_preproc_fields:
            preproc[fld]        = self.load
        # fields that need some form of preprocessing
        preproc['tcm']          = self.calc_tcm
        preproc['IFS']          = self.calc_IFS_gust_term
        preproc['wdir']         = self.calc_wdir
        preproc['zbralb']       = self.calc_zbra
        preproc['zbraes']       = self.calc_zbra
        preproc['zbraub']       = self.calc_zbra
        preproc['sso_stdh']     = self.load_external
        preproc['hsurf']        = self.load_external
        self.preproc = preproc

        ps = {}
        self.predictor_structure = ps

        ps['tcm']               =   {'fix':0,
                                    'prod':[('tcm',1)]
                                    }
        ps['tke']               =   {'fix':0,
                                    'prod':[('tkel1',1)]
                                    }
        ps['IFS']               =   {'fix':0,
                                    'prod':[('IFS',1)]
                                    }
        ps['IFS_2']             =   {'fix':0,
                                    'prod':[('IFS',2)]
                                    }
        ps['IFS_3']             =   {'fix':0,
                                    'prod':[('IFS',3)]
                                    }
        ps['IFSfix']            =   {'fix':1,
                                    'prod':[('IFS',1)]
                                    }
        #######################################################################
        ###### zvp10
        #######################################################################
        ps['zvp10fix']          =   {'fix':1,
                                    'prod':[('zvp10',1)]
                                    }
        ps['zvp10']             =   {'fix':0,
                                    'prod':[('zvp10',1)]
                                    }
        ps['zvp10_2']           =   {'fix':0,
                                    'prod':[('zvp10',2)]
                                    }
        ps['zvp10_3']           =   {'fix':0,
                                    'prod':[('zvp10',3)]
                                    }
        ps['(zvp10)']           =   {'fix':0,'transform':'log',
                                    'prod':[('zvp10',1)]
                                    }
        ps['(zvp10_tcm)']       =   {'fix':0,'transform':'log',
                                    'prod':[('zvp10',1),('tcm',1)]
                                    }
        ps['zvp10_tcm']         =   {'fix':0,
                                    'prod':[('zvp10',1),('tcm',1)]
                                    }
        ps['zvp10_2_tcm']       =   {'fix':0,
                                    'prod':[('zvp10',2),('tcm',1)]
                                    }
        ps['zvp10_wdir']        =   {'fix':0,
                                    'prod':[('zvp10',1),('wdir',1)]
                                    }
        ps['zvp10_tke']         =   {'fix':0,
                                    'prod':[('zvp10',1),('tkel1',1)]
                                    }
        ps['zvp10_IFS']         =   {'fix':0,
                                    'prod':[('zvp10',1),('IFS',1)]
                                    }
        ps['zvp10_IFS_2']       =   {'fix':0,
                                    'prod':[('zvp10',1),('IFS',2)]
                                    }
        ps['zvp10_IFS_3']       =   {'fix':0,
                                    'prod':[('zvp10',1),('IFS',3)]
                                    }
        ps['zvp10_IFS_4']       =   {'fix':0,
                                    'prod':[('zvp10',1),('IFS',4)]
                                    }
        ps['zvp10_z0']          =   {'fix':0,
                                    'prod':[('zvp10',1),('z0',1)]
                                    }
        ps['zvp10_sso']         =   {'fix':0,
                                    'prod':[('zvp10',1),('sso_stdh',1)]
                                    }
        ps['zvp10_sso_2']       =   {'fix':0,
                                    'prod':[('zvp10',1),('sso_stdh',2)]
                                    }
        ps['zvp10_sso_3']       =   {'fix':0,
                                    'prod':[('zvp10',1),('sso_stdh',3)]
                                    }
        ps['zvp10_hsurf']       =   {'fix':0,
                                    'prod':[('zvp10',1),('hsurf',1)]
                                    }
        ps['zvp10_hsurf_2']     =   {'fix':0,
                                    'prod':[('zvp10',1),('hsurf',2)]
                                    }
        ps['zvp10_hsurf_3']     =   {'fix':0,
                                    'prod':[('zvp10',1),('hsurf',3)]
                                    }
        #######################################################################
        ###### bralb
        #######################################################################
        ps['zvp10_bralb']       =   {'fix':0,
                                    'prod':[('zvp10',1),('zv_bra_lb',1)]
                                    }
        ps['zvp10_bralb_2']     =   {'fix':0,
                                    'prod':[('zvp10',1),('zv_bra_lb',2)]
                                    }
        ps['zvp10_bralb_3']     =   {'fix':0,
                                    'prod':[('zvp10',1),('zv_bra_lb',3)]
                                    }
        ps['zvp10_bralb_4']    =   {'fix':0,
                                    'prod':[('zvp10',1),('zv_bra_lb',4)]
                                    }
        ps['zbralb']            =   {'fix':0,
                                    'prod':[('zbralb',1)]
                                    }
        ps['zvp10_zbralb']      =   {'fix':0,
                                    'prod':[('zvp10',1),('zbralb',1)]
                                    }
        ps['zvp10_zbralb_2']    =   {'fix':0,
                                    'prod':[('zvp10',1),('zbralb',2)]
                                    }
        ps['zvp10_zbralb_3']    =   {'fix':0,
                                    'prod':[('zvp10',1),('zbralb',3)]
                                    }
        #######################################################################
        ###### braes
        #######################################################################
        ps['zvp10_braes']       =   {'fix':0,
                                    'prod':[('zvp10',1),('zv_bra_es',1)]
                                    }
        ps['zvp10_braes_2']     =   {'fix':0,
                                    'prod':[('zvp10',1),('zv_bra_es',2)]
                                    }
        ps['zvp10_braes_3']     =   {'fix':0,
                                    'prod':[('zvp10',1),('zv_bra_es',3)]
                                    }
        ps['zbraes']            =   {'fix':0,
                                    'prod':[('zbraes',1)]
                                    }
        ps['zvp10_zbraes']      =   {'fix':0,
                                    'prod':[('zvp10',1),('zbraes',1)]
                                    }
        ps['zvp10_zbraes_2']    =   {'fix':0,
                                    'prod':[('zvp10',1),('zbraes',2)]
                                    }
        ps['zvp10_zbraes_3']    =   {'fix':0,
                                    'prod':[('zvp10',1),('zbraes',3)]
                                    }
        #######################################################################
        ###### braub
        #######################################################################
        ps['zvp10_braub']       =   {'fix':0,
                                    'prod':[('zvp10',1),('zv_bra_ub',1)]
                                    }
        ps['zvp10_braub_2']     =   {'fix':0,
                                    'prod':[('zvp10',1),('zv_bra_ub',2)]
                                    }
        ps['zvp10_braub_3']     =   {'fix':0,
                                    'prod':[('zvp10',1),('zv_bra_ub',3)]
                                    }
        ps['zbraub']            =   {'fix':0,
                                    'prod':[('zbraub',1)]
                                    }
        ps['zvp10_zbraub']      =   {'fix':0,
                                    'prod':[('zvp10',1),('zbraub',1)]
                                    }


    def load(self, field_name):
        print('load ' + str(field_name))
        field_values = np.ma.filled(self.ncf[field_name][:], fill_value=np.nan)
        # set array_shape attribute which can be use for
        # external fields later
        if self.array_shape is None:
            self.array_shape = field_values.shape
        return(field_values)

    def calc_tcm(self, field_name):
        tcm = self.load(field_name)
        tcm[tcm < 5E-4] = 5E-4
        tcm = np.sqrt(tcm)
        return(tcm)

    def calc_IFS_gust_term(self, field_name):

        ugn = 1
        #ugn = 7.71
        #ugn *= 3
        hpbl = 1000
        g = 9.80665
        Rd = 287.05
        etv = 0.6078
        cp = 1005.0
        kappa = Rd/cp

        umlev = self.load('ul1')
        vmlev = self.load('vl1')
        ps = self.load('ps')
        qvflx = self.load('qvflx')
        shflx = self.load('shflx')
        z0 = self.load('z0')
        Tskin = self.load('Tskin')
        Tmlev = self.load('Tl1')
        qvmlev = self.load('qvl1')
        phimlev = self.load('phil1')

        # density
        rho = ps / ( Rd*Tmlev * ( 1 + etv*qvmlev ) )

        # buoyancy
        buoy = g * ( - etv*qvflx - shflx/( Tskin*cp ) ) / rho

        # surface stress
        zcdn = ( kappa / np.log( 1 + phimlev/(g*z0) ) )**2
        dua = np.sqrt( np.maximum( 0.1**2, umlev**2 + vmlev**2 ) )
        ustr = rho*umlev*dua*zcdn
        vstr = rho*vmlev*dua*zcdn

        # friction velocity
        ustar2 = np.sqrt( ustr**2 + vstr**2 ) / rho 
        wstar2 = ( buoy[buoy > 0]*hpbl )**(2/3)
        ustar2[buoy > 0] = ustar2[buoy > 0] + 2E-3*wstar2
        ustar = np.maximum( np.sqrt(ustar2), 0.0001 )

        # wind gust
        idl = -hpbl*kappa*buoy/ustar**3
        gust_term = np.zeros( idl.shape )
        greater0 = idl >= 0
        gust_term[greater0] = ustar[greater0]*ugn
        smaller0 = idl < 0
        gust_term[smaller0] = ustar[smaller0]*ugn * (1 - 0.5/12*idl[smaller0])**(1/3)

        return(gust_term)

    def calc_zbra(self, field_name):
        bra_type = field_name[-2:]
        if bra_type == 'es':
            kbra_load_name = 'k_bra_es'
        elif bra_type == 'lb':
            kbra_load_name = 'k_bra_lb'
        elif bra_type == 'ub':
            kbra_load_name = 'k_bra_ub'
        else:
            raise ValueError()
        kbra = self.load(kbra_load_name)
        zbra = np.copy(kbra)
        kalts = np.loadtxt('../data/kaltitudes.txt')
        kinds = kalts[:,0].astype(np.int)
        kalts = kalts[:,1]
        for i,kind in enumerate(kinds):
            zbra[kbra == kind] = kalts[i]
        return(zbra)


    def calc_wdir(self, field_name):

        wdir_shift = np.pi*1/4

        umlev = self.load('ul1')
        vmlev = self.load('vl1')

        wind_abs = np.sqrt(umlev**2 + vmlev**2)
        wind_dir_trig_to = np.arctan2(umlev/wind_abs, vmlev/wind_abs)

        wdir = np.sin(wind_dir_trig_to + wdir_shift)

        return(wdir)


    def load_external(self, field_name):
        
        if self.array_shape is None:
            raise ValueError('Predictors class does not know about ' +\
                    'array shape. Load another (non-external) field first.')

        nhrs_forecast   = self.array_shape[0]
        nstat           = self.array_shape[1]
        n_ts_per_hr     = self.array_shape[2]
        ext_field = np.full( self.array_shape , np.nan)
        for sI,stat_name in enumerate(self.data[G.STAT_NAMES]):
            #stat_name = 'ABO'
            if stat_name != 'nan':
                ext_field[:,sI,:] = self.data[G.STAT_META]\
                                    [stat_name][field_name].values[0]

        return(ext_field)

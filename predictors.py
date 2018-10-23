import numpy as np





class Predictors:

    def __init__(self, ncf):
        
        # nc file containing all the fields
        self.ncf = ncf

        preproc = {}
        no_preproc_fields = ['zvp10', 'zv_bra_lb', 'zv_bra_es', 'zv_bra_ub']
        for fld in no_preproc_fields:
            preproc[fld]        = self.load
        preproc['tcm']          = self.calc_tcm
        preproc['IFS']          = self.calc_IFS_gust_term
        self.preproc = preproc



    def load(self, field_name):
        #print('load ' + str(field_name))
        field_values = np.ma.filled(self.ncf[field_name][:], fill_value=np.nan)
        return(field_values)

    def calc_tcm(self, field_name):
        tcm = self.load(field_name)
        tcm[tcm < 5E-4] = 5E-4
        tcm = np.sqrt(tcm)
        return(tcm)

    def calc_IFS_gust_term(self, field_name):

        ugn = 7.71
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

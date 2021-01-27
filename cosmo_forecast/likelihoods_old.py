from lya_bao.models import baoModel
from lya_bao.interpolator import baoInterpolator
from numba import jit
import scipy.constants as const
import numpy as np


class baoLikelihood:
    '''
    Class for computing BAO likelihood using a gaussian likelihood
    See the example ini file for setup of config
    '''
    def __init__(self, config):
        # Get Selection list
        sel_str = config['Likelihood'].get('data')
        sel = [x.strip() for x in sel_str.split(',')]

        self.zeff_dict = {}
        # Call the right setup functions for each data set
        if 'Galaxies' in sel:
            gal_zeff = self.setup_gal(config['Galaxies'])
            self.zeff_dict['Galaxies'] = np.array(gal_zeff)
        if 'Ly-alpha' in sel:
            lya_zeff = self.setup_lya(config['Ly-alpha'])
            self.zeff_dict['Ly-alpha'] = np.array(lya_zeff)
        if 'Forecast' in sel:
            self.forecastLik = baoForecastLikelihood(config)
            self.zeff_dict['Forecast'] = np.array(self.forecastLik.zeff_list)
        if 'Omega_b' in sel:
            self.setup_omb(config['Omega_b'])
        if 'H_0' in sel:
            self.setup_h0(config['H_0'])
        
        self.select_data = sel
        
        if len(self.zeff_dict.keys()) is 0:
            raise RuntimeError

        # Build a matrix with model objects
        # self.model_dict = {}
        # for key in zeff_dict.keys():
        #     self.model_dict[key] = []
        #     for zeff in zeff_dict[key]:
        #         self.model_dict[key].append(baoModel(z_eff = zeff))

        self.compute_rd = config['Model'].getboolean('compute_rd')
        self.model = config['Model'].get('model')

    @jit
    def log_lik(self, theta):
        '''
        Compute log Likelihood from BAO data 
        '''
        model_pars, derived = self.compute_model(theta)

        log_lik = 0

        if 'Galaxies' in self.select_data:
            gal_pars = model_pars['Galaxies'].T
            log_lik += self.compute_galaxies_lik(gal_pars)
        
        if 'Ly-alpha' in self.select_data:
            lya_pars = model_pars['Ly-alpha'].T
            log_lik += self.compute_lya_interp_lik(lya_pars)
        
        if 'Forecast' in self.select_data:
            forecast_pars = model_pars['Forecast'].T
            log_lik += self.forecastLik.log_lik(forecast_pars)

        if 'Omega_b' in self.select_data:
            assert(self.compute_rd)
            omb_th = theta[2] * (theta[1]/100)**2
            log_lik += self.gaussian_lik(omb_th, self.omb_data['mean'], self.omb_data['sigma'])

        if 'H_0' in self.select_data:
            assert(self.compute_rd)
            log_lik += self.gaussian_lik(theta[1], self.h0_data['mean'], self.h0_data['sigma'])

        return log_lik, derived

    @jit
    def compute_model(self, theta):
        '''
        Computes the model parameters for everything using the zeff_dict member
        theta must be of the form specified by the model

        flcdm -> theta = ['Omc', 'H_0 r_d']
        lcdm -> theta = ['Omc', 'H_0 r_d', 'Ode']
        fwcdm -> theta = ['Omc', 'H_0 r_d', 'w0]
        wcdm -> theta = ['Omc', 'H_0 r_d', 'Ode', 'w0']

        Omc (Omega_cdm) is only used to compute Omm
        If compute_rd is True replace 'H_0 r_d' with two parameters:
        -> 'H_0', 'Omb' in all of the above

        '''
        # Check for Omega_b and H_0
        if self.compute_rd:
            H0 = theta[1]
            Omb = theta[2]
            Omm = theta[0] + Omb
        else:
            Omm = theta[0]
            H0 = None
            Omb = None

        # Get the remaining model parameters
        if self.model == 'flcdm':
            assert(len(theta) == 2 + int(self.compute_rd))
            Ode = None
            w0 = -1
        elif self.model == 'lcdm':
            assert(len(theta) == 3 + int(self.compute_rd))
            Ode = theta[-1]
            w0 = -1
        elif self.model == 'fwcdm':
            assert(len(theta) == 3 + int(self.compute_rd))
            Ode = None
            w0 = theta[-1]
        elif self.model == 'wcdm':
            assert(len(theta) == 4 + int(self.compute_rd))
            Ode = theta[-2]
            w0 = theta[-1]
        else:
            print('Model.model must be one of: ["flcdm","lcdm","fwcdm","wcdm"].'  
                'The ini file value was: %s. Please change and rerun' % self.model)
            raise ValueError

        bao_model = baoModel(Omm, H0, Omb, Ode, w0)

        pars = {}
        for key in self.zeff_dict.keys():
            if self.compute_rd:
                pars[key] = bao_model.compute_anchored(self.zeff_dict[key])
            else:
                H0_rd = theta[1]
                pars[key] = bao_model.compute(self.zeff_dict[key], H0_rd)

        derived = bao_model.derived
        return pars, derived

    def setup_gal(self, config):
        '''
        Setup the things we need to compute a likelihood from Galaxies BAO
        For now this works with:
        - BOSS DR12
        - 6DF
        - SDSS MGS
        '''
        self.gal_data = {}

        # Figure out which data sets we need
        self.gal_data['boss'] = config.getboolean('use_boss')
        self.gal_data['6df'] = config.getboolean('use_6df')
        self.gal_data['mgs'] = config.getboolean('use_mgs')
        self.gal_data['lrg'] = config.getboolean('use_lrg')
        self.gal_data['qso'] = config.getboolean('use_qso')

        gal_zeff = []

        # BOSS first
        if self.gal_data['boss']:
            data = np.loadtxt(config.get('boss_data'))
            cov = np.loadtxt(config.get('boss_cov'))

            boss_zeff_list = data[:, 0]
            data_vec = data[:, 1:].flatten() 
            assert(len(data_vec) == 2*len(boss_zeff_list))
            assert(len(data_vec) == len(cov))
            assert(len(cov) == len(cov.T))
            
            self.gal_data['boss_data_vec'] = data_vec
            self.gal_data['boss_cov_mat'] = cov
            self.gal_data['boss_rd_fid'] = config.getfloat('boss_rd_fid')

            gal_zeff = list(boss_zeff_list)

        # 6DF
        if self.gal_data['6df']:
            str_6df = config.get('data_6df')
            data_6df = [float(x.strip()) for x in str_6df.split(',')]
            # This contains mean,sigma
            self.gal_data['6df_data'] = data_6df
            zeff_6df = config.getfloat('zeff_6df')
            self.gal_data['6df_zeff'] = zeff_6df
            gal_zeff.append(zeff_6df)

        # SDSS MGS
        if self.gal_data['mgs']:
            str_mgs = config.get('data_mgs')
            data_mgs = [float(x.strip()) for x in str_mgs.split(',')]
            # This contains mean,sigma
            self.gal_data['mgs_data'] = data_mgs
            zeff_mgs = config.getfloat('zeff_mgs')
            self.gal_data['mgs_zeff'] = zeff_mgs
            gal_zeff.append(zeff_mgs)

        # eBOSS LRGs
        if self.gal_data['lrg']:
            str_lrg = config.get('data_lrg')
            data_lrg = [float(x.strip()) for x in str_lrg.split(',')]
            # This contains mean,sigma
            self.gal_data['lrg_data'] = data_lrg
            zeff_lrg = config.getfloat('zeff_lrg')
            self.gal_data['lrg_zeff'] = zeff_lrg
            gal_zeff.append(zeff_lrg)

        # eBOSS QSOs
        if self.gal_data['qso']:
            str_qso = config.get('data_qso')
            data_qso = [float(x.strip()) for x in str_qso.split(',')]
            # This contains mean,sigma
            self.gal_data['qso_data'] = data_qso
            zeff_qso = config.getfloat('zeff_qso')
            self.gal_data['qso_zeff'] = zeff_qso
            gal_zeff.append(zeff_qso)

        return gal_zeff

    def setup_lya(self, config):
        '''
        Setup the things we need to compute a likelihood from Ly-alpha BAO
        For now this uses chisq tables from BOSS DR11
        '''
        num_files = config.getint('num_files')
        self.bao_interp = baoInterpolator(config)
        
        zeff_lya = []
        for i in range(num_files):
            zeff_lya.append(config.getfloat('zeff_' + str(i)))
        
        self.lya_data = {}
        self.lya_data['zeff_list'] = zeff_lya

        return zeff_lya

    def setup_omb(self, config):
        '''
        Omega_b likelihood parameters
        '''
        self.omb_data = {}
        self.omb_data['mean'] = config.getfloat('omega_b_mean')
        self.omb_data['sigma'] = config.getfloat('omega_b_sig')

    def setup_h0(self, config):
        '''
        H_0 likelihood parameters
        '''
        self.h0_data = {}
        self.h0_data['mean'] = config.getfloat('H_0_mean')
        self.h0_data['sigma'] = config.getfloat('H_0_sig')
    
    @jit
    def compute_lya_interp_lik(self, model_pars):
        '''
        Compute Lya BAO likelihood using
        interpolation on the chisq table
        We get:
        D_M/rd and H*rd
        We send to the interpolators:
        D_A/rd and c/(H*rd)
        '''
        assert(self.bao_interp.num_interp == len(model_pars))

        da_rd_vec = []
        c_H0rd_vec = []
        for i in range(len(model_pars)):
            da_rd = model_pars[i][0] / (1 + self.lya_data['zeff_list'][i])
            c_H0rd = (const.c / 1000) / model_pars[i][1]

            da_rd_vec.append(da_rd)
            c_H0rd_vec.append(c_H0rd)

        # compute log lik (not normalized)
        log_lik = -0.5 * self.bao_interp.compute_chisq(da_rd_vec, c_H0rd_vec)
        
        return log_lik

    @jit
    def compute_galaxies_lik(self, model_pars):
        '''
        Compute Galaxy BAO likelihood 
        We get: [(D_M(z) / r_d), (H(z) * r_d)]

        Parameters used are:
        BOSS:
        -> D_M(z_eff) * rd_fid / rd
        -> [H(z_eff) * rd] / rd_fid

        6df:
        -> r_d / D_V

        mgs, lrg and qso:
        -> D_V / r_d
        '''
        log_lik = 0
        pos = 0 # position in model_pars vector

        # BOSS first
        if self.gal_data['boss']:
            pars = model_pars[:3,:]
            pos += 3
            # compute the right parameters specified above
            pars.T[0] = pars.T[0] * self.gal_data['boss_rd_fid']
            pars.T[1] = pars.T[1] / self.gal_data['boss_rd_fid']
            
            # build the model vector
            model_vector = pars.flatten()
            assert(len(model_vector) == len(self.gal_data['boss_data_vec']))

            log_lik += self.linalg_lik(model_vector, self.gal_data['boss_data_vec'], self.gal_data['boss_cov_mat'])
        
        # 6DF
        if self.gal_data['6df']:
            # -> rd / D_v
            pars_6df = model_pars[pos,:]
            pos += 1

            data_6df = self.gal_data['6df_data'][0]
            sig_6df = self.gal_data['6df_data'][1]
            model_6df = ((const.c/1000) * self.gal_data['6df_zeff'] * pars_6df[0]**2 / pars_6df[1])**(-1.0/3.0)
            log_lik -= 0.5 * np.log(2 * np.pi) - np.log(sig_6df)
            log_lik -= 0.5 * (model_6df - data_6df)**2 / sig_6df**2

        # SDSS MGS
        if self.gal_data['mgs']:
            # -> D_v / rd
            pars_mgs = model_pars[pos,:]
            pos += 1

            data_mgs = self.gal_data['mgs_data'][0]
            sig_mgs = self.gal_data['mgs_data'][1]
            model_mgs = ((const.c/1000) * self.gal_data['mgs_zeff'] * pars_mgs[0]**2 / pars_mgs[1])**(1.0/3.0)
            log_lik -= 0.5 * np.log(2 * np.pi) - np.log(sig_mgs)
            log_lik -= 0.5 * (model_mgs - data_mgs)**2 / sig_mgs**2

        # eBOSS LRGs
        if self.gal_data['lrg']:
            # -> D_v / rd
            pars_lrg = model_pars[pos,:]
            pos += 1

            data_lrg = self.gal_data['lrg_data'][0]
            sig_lrg = self.gal_data['lrg_data'][1]
            model_lrg = ((const.c/1000) * self.gal_data['lrg_zeff'] * pars_lrg[0]**2 / pars_lrg[1])**(1.0/3.0)
            log_lik -= 0.5 * np.log(2 * np.pi) - np.log(sig_lrg)
            log_lik -= 0.5 * (model_lrg - data_lrg)**2 / sig_lrg**2

        # eBOSS QSOs
        if self.gal_data['qso']:
            # -> D_v / rd
            pars_qso = model_pars[pos,:]
            pos += 1

            data_qso = self.gal_data['qso_data'][0]
            sig_qso = self.gal_data['qso_data'][1]
            model_qso = ((const.c/1000) * self.gal_data['qso_zeff'] * pars_qso[0]**2 / pars_qso[1])**(1.0/3.0)
            log_lik -= 0.5 * np.log(2 * np.pi) - np.log(sig_qso)
            log_lik -= 0.5 * (model_qso - data_qso)**2 / sig_qso**2
        
        return log_lik

    @staticmethod
    @jit
    def linalg_lik(model_vector, data_vec, cov_mat_ar):
        # Compute the stuff we need fot the likelihood
        num_dim = len(model_vector)
        cov_mat = np.matrix(cov_mat_ar)
        cov_det = np.linalg.det(cov_mat)
        cov_inv = np.linalg.inv(cov_mat)
        
        # Gaussian Likelihood
        diff_vec = model_vector - data_vec
        chisq = cov_inv.dot(diff_vec)
        chisq = float(diff_vec * chisq.T)
        
        log_lik = -0.5 * num_dim * np.log(2 * np.pi) - 0.5 * np.log(cov_det)
        log_lik -= 0.5 * chisq
        return log_lik

    @staticmethod
    @jit
    def gaussian_lik(th, mean, sig):
        log_lik = -0.5 * np.log(2 * np.pi) - np.log(sig)
        log_lik -= 0.5 * (th - mean)**2 / sig**2
        return log_lik

    # @staticmethod
    # @jit
    # def compute_lya_gaussian_lik(model_params_list, z_eff, data_vec, cov_mat_ar):
    #     '''
    #     Compute Lya BAO likelihood
    #     Parameters used are: 
    #     -> D_A(z_eff) / rd
    #     -> c / [H(z_eff) * rd]
    #     '''
    #     pars = model_params_list.copy()
        
    #     # compute the right parameters specified above
    #     pars.T[0] = pars.T[0] / (1 + z_eff)
    #     pars.T[1] = (const.c/1000.0) /pars.T[1]
        
    #     # build the model vector
    #     model_vector = pars.flatten()
    #     assert(len(model_vector) == len(data_vec))

    #     # Compute the stuff we need for the likelihood
    #     num_dim = len(model_vector)
    #     cov_mat = np.matrix(cov_mat_ar)
    #     cov_det = np.linalg.det(cov_mat)
    #     cov_inv = np.linalg.inv(cov_mat)
        
    #     # Gaussian Likelihood
    #     diff_vec = model_vector - data_vec
    #     chisq = cov_inv.dot(diff_vec)
    #     chisq = float(diff_vec * chisq.T)
        
    #     log_lik = -0.5 * num_dim * np.log(2 * np.pi) - 0.5 * np.log(cov_det)
    #     log_lik -= 0.5 * chisq
    #     return log_lik


class baoForecastLikelihood:
    '''
    Likelihood for computing a forecast (for DESI) 
    Needs tables with -- redshift -- percent error in (D_A/rd) -- percent error in (H*rd)

    For now everything will be treated the same (both Gal and Lya)
    '''
    def __init__(self, config):
        self.num_files = config['Forecast'].getint('num_files')

        # First get Fiducial cosmology
        fid_H0 = config['Model'].getfloat('fid_H0', 67.37)
        fid_Omm = config['Model'].getfloat('fid_Omega_m', 0.3147)
        fid_Omb = config['Model'].getfloat('fid_Omega_b', 0.049199)
        corr_par = config['Model'].getfloat('corr_par', 0.4)

        self.zeff_list = [] # coords: num_files, bin - each contains one zeff
        self._data = [] # coords: num_files, bin - each contains one dict with data

        for i in range(self.num_files):
            path = config['Forecast'].get('data_' + str(i))
            data = np.loadtxt(path)
            if len(np.shape(data)) is 1:
                data = np.array([data])

            # Get zeff
            zeff = list(data[:,0])
            self.zeff_list += zeff

            data_list = []
            # Compute the fiducial models
            for row in data:
                # Compute fiducial data
                model = baoModel(row[0])
                pars = model.compute_with_H0_approx(fid_H0, fid_Omm, fid_Omb)
                pars[0] = pars[0] / (1 + row[0]) # convert D_M -> D_A
                pars = np.array(pars)

                # Compute errors and the covariance matrix
                sig_1 = pars[0] * row[1] / 100.0
                sig_2 = pars[1] * row[2] / 100.0
                cov = corr_par * sig_1 * sig_2
                cov_mat = np.matrix([[sig_1**2, cov], [cov, sig_2**2]])
                inv_cov = np.linalg.inv(cov_mat)
                cov_det = np.linalg.det(cov_mat)

                # Save everything we need in a dict
                data_dict = {}
                data_dict['zeff'] = row[0]
                data_dict['data_vec'] = pars
                data_dict['cov_mat'] = cov_mat
                data_dict['inv_cov'] = inv_cov
                data_dict['cov_det'] = cov_det

                data_list.append(data_dict)

            self._data.append(data_list)

    @jit
    def log_lik(self, theta):
        '''
        theta must contain a table of BAO peak position parameters
        [(D_M(z) / r_d), (H(z) * r_d)]
        Matching the zeff_list member variable length
        '''
        log_lik = 0
        assert len(self.zeff_list) == len(theta)

        ind = 0  # index for theta vec
        for i, probe in enumerate(self._data):
            for j, data_dict in enumerate(probe):
                th_vec = theta[ind]
                assert len(th_vec) == 2
                th_vec[0] = th_vec[0] / (1 + data_dict['zeff'])

                log_lik += self.comp_lik(th_vec, data_dict['data_vec'], data_dict['inv_cov'], data_dict['cov_det'])
                ind += 1

        return log_lik

    @staticmethod
    @jit
    def comp_lik(th_vec, data_vec, inv_cov, cov_det):
        '''
        Compute N-D gaussian likelihood
        '''
        num_dim = len(th_vec)

        # Gaussian Likelihood
        diff_vec = th_vec - data_vec
        chisq = inv_cov.dot(diff_vec)
        chisq = float(diff_vec * chisq.T)

        log_lik = -0.5 * num_dim * np.log(2 * np.pi) - 0.5 * np.log(cov_det)
        log_lik -= 0.5 * chisq
        return log_lik

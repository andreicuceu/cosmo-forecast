from cosmo_forecast.models import baoModel
import numpy as np


class baoLikelihood:
    '''
    Class for computing BAO likelihood using a gaussian likelihood
    See the example ini file for setup of config
    '''
    F_ap = {}
    sig_F_ap = {}
    alpha = {}
    sig_alpha = {}
    gamma = {}
    sig_gamma = {}
    bao_data = {}
    bao_cov_inv = {}

    def __init__(self, config):
        # Get Selection list
        selection_str = config['Likelihood'].get('data')
        datasets = [x.strip() for x in selection_str.split(',')]

        self.zeff_dict = {}
        for dataname in datasets:
            if 'AP' in dataname:
                self.setup_ap(config[dataname], dataname)
            elif 'alpha' in dataname:
                self.setup_alpha(config[dataname], dataname)
            elif 'gamma' in dataname:
                self.setup_gamma(config[dataname], dataname)
            elif 'BAO' in dataname:
                self.setup_bao(config[dataname], dataname)
            elif 'DESI' in dataname:
                self.setup_desi(config[dataname], dataname)
            else:
                raise ValueError('Wrong dataset name')

            if 'DESI' in dataname:
                self.zeff_dict[dataname] = self.desi_zeff_list
            else:
                self.zeff_dict[dataname] = config[dataname].getfloat('redshift')

        self.datasets = datasets
        self.compute_rd = config['Model'].getboolean('compute_rd', False)
        self.model = config['Model'].get('model')

    # @jit
    def log_lik(self, theta):
        '''
        Compute log Likelihood from BAO data
        '''
        model_pars, derived = self.compute_model(theta)

        log_lik = 0

        for name in self.datasets:
            if 'AP' in name:
                log_lik += self.lik_ap(model_pars[name], name)
            if 'alpha' in name:
                log_lik += self.lik_alpha(model_pars[name], name)
            if 'gamma' in name:
                log_lik += self.lik_gamma(model_pars[name], name)
            if 'BAO' in name:
                log_lik += self.lik_bao(model_pars[name], name)
            if 'DESI' in name:
                log_lik += self.lik_desi(model_pars[name])

        return log_lik, derived

    # @jit
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
            raise ValueError('compute_rd not implemented properly yet')
            H0 = theta['h0']
            Omb = theta['omega_b']
            Omm = theta['omega_cdm'] + Omb
        else:
            Omm = theta['omega_m']
            if 'h0_rd' in theta.keys():
                H0_rd = theta['h0_rd']
            H0 = None
            Omb = None

        # Get the remaining model parameters
        if self.model == 'flcdm':
            Ode = None
            w0 = -1
        elif self.model == 'lcdm':
            Ode = theta['omega_lambda']
            w0 = -1
        elif self.model == 'fwcdm':
            Ode = None
            w0 = theta['w0']
        elif self.model == 'wcdm':
            Ode = theta['omega_lambda']
            w0 = theta['w0']
        else:
            print('Model.model must be one of: ["flcdm","lcdm","fwcdm","wcdm"].'  
                'The ini file value was: %s. Please change and rerun' % self.model)
            raise ValueError

        bao_model = baoModel(Omm, H0, Omb, Ode, w0)

        pars = {}
        for name, zeff in self.zeff_dict.items():
            if 'AP' in name:
                pars[name] = bao_model.compute_ap(zeff)
            if 'alpha' in name:
                pars[name] = bao_model.compute_alpha(zeff, H0_rd)
            if 'gamma' in name:
                pars[name] = bao_model.compute_gamma(zeff)
            if 'BAO' in name:
                ap = bao_model.compute_ap(zeff)
                alpha = bao_model.compute_alpha(zeff, H0_rd)
                pars[name] = np.array([alpha, ap])
            if 'DESI' in name:
                pars[name] = bao_model.compute_anchored(np.array(zeff)).T

        derived = bao_model.derived
        if derived is None:
            derived = []
        return pars, derived

    def setup_ap(self, config, name):
        self.F_ap[name] = config.getfloat('F_ap')
        self.sig_F_ap[name] = config.getfloat('sig_F_ap')

    def setup_alpha(self, config, name):
        self.alpha[name] = config.getfloat('alpha')
        self.sig_alpha[name] = config.getfloat('sig_alpha')

    def setup_gamma(self, config, name):
        self.gamma[name] = config.getfloat('gamma')
        self.sig_gamma[name] = config.getfloat('sig_gamma')

    def setup_bao(self, config, name):
        ap = config.getfloat('F_ap')
        alpha = config.getfloat('alpha')
        sig_ap = config.getfloat('sig_F_ap')
        sig_alpha = config.getfloat('sig_alpha')
        rho = config.getfloat('rho_bao')
        corr = rho * sig_alpha * sig_ap
        cov = np.array([[sig_alpha**2, corr], [corr, sig_ap**2]])

        self.bao_data[name] = np.array([alpha, ap])
        self.bao_cov_inv[name] = np.linalg.inv(cov)

    def setup_desi(self, config, name):
        self.desi_num_files = config['DESI'].getint('num_files')

        # First get Fiducial cosmology
        fid_H0 = config['Model'].getfloat('fid_H0', 67.37)
        fid_Omm = config['Model'].getfloat('fid_Omega_m', 0.3147)
        fid_Omb = config['Model'].getfloat('fid_Omega_b', 0.049199)
        corr_par = config['Model'].getfloat('corr_par', 0.4)

        self.desi_zeff_list = []  # coords: num_files, bin - each contains one zeff
        self.desi_data = []  # coords: num_files, bin - each contains one dict with data

        for i in range(self.desi_num_files):
            path = config['DESI'].get('data_' + str(i))
            data = np.loadtxt(path)
            if len(np.shape(data)) == 1:
                data = np.array([data])

            # Get zeff
            zeff = list(data[:, 0])
            self.zeff_list += zeff

            data_list = []
            # Compute the fiducial models
            for row in data:
                # Compute fiducial data

                model = baoModel(fid_Omm, fid_H0, fid_Omb)
                pars = model.compute_anchored(row[0])
                pars[0] = pars[0] / (1 + row[0])  # convert D_M -> D_A

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

            self.desi_data.append(data_list)

    def lik_ap(self, dm_dh, name):
        chi2 = (dm_dh - self.F_ap[name])**2 / self.sig_F_ap[name]**2
        return -chi2 / 2

    def lik_alpha(self, dmdh_rdsq, name):
        chi2 = (dmdh_rdsq - self.alpha[name])**2 / self.sig_alpha[name]**2
        return -chi2 / 2

    def lik_gamma(self, dmdh, name):
        chi2 = (dmdh - self.gamma[name])**2 / self.sig_gamma[name]**2
        return -chi2 / 2

    def lik_bao(self, bao_model, name):
        diff_vec = bao_model - self.bao_data[name]
        chisq = self.bao_cov_inv[name].dot(diff_vec)
        chisq = float(diff_vec.T.dot(chisq))
        return -chisq / 2

    def lik_desi(self, theta):
        '''
        theta must contain a table of BAO peak position parameters
        [(D_M(z) / r_d), (H(z) * r_d)]
        Matching the zeff_list member variable length
        '''
        log_lik = 0
        assert len(self.desi_zeff_list) == len(theta)

        ind = 0  # index for theta vec
        for i, probe in enumerate(self.desi_data):
            for j, data_dict in enumerate(probe):
                th_vec = theta[ind]
                assert len(th_vec) == 2
                th_vec[0] = th_vec[0] / (1 + data_dict['zeff'])

                log_lik += self.comp_lik(th_vec, data_dict['data_vec'], data_dict['inv_cov'], data_dict['cov_det'])
                ind += 1

        return log_lik

    @staticmethod
    def comp_lik(th_vec, data_vec, inv_cov, cov_det):
        '''
        Compute N-D gaussian likelihood
        '''
        # num_dim = len(th_vec)

        # Gaussian Likelihood
        diff_vec = th_vec - data_vec
        chisq = inv_cov.dot(diff_vec)
        chisq = float(diff_vec.T.dot(chisq))

        # log_lik = -0.5 * num_dim * np.log(2 * np.pi) - 0.5 * np.log(cov_det)
        # log_lik -= 0.5 * chisq
        return -chisq / 2

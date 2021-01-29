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
            else:
                raise ValueError('Wrong dataset name')

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
        chisq = float(diff_vec * chisq.T)
        return -chisq / 2

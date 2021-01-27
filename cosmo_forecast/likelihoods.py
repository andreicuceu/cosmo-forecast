from cosmo_forecast.models import baoModel


class baoLikelihood:
    '''
    Class for computing BAO likelihood using a gaussian likelihood
    See the example ini file for setup of config
    '''
    F_ap = {}
    sig_F_ap = {}

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
            raise ValueError('compute_rd not implemented properly yet')
            H0 = theta[1]
            Omb = theta[2]
            Omm = theta[0] + Omb
        else:
            Omm = theta[0]
            H0_rd = theta[1]
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
        for name, zeff in self.zeff_dict.items():
            if 'AP' in name:
                pars[name] = bao_model.compute_ap(zeff)
            if 'alpha' in name:
                pars[name] = bao_model.compute_alpha(zeff, H0_rd)
            if 'gamma' in name:
                pars[name] = bao_model.compute_gamma(zeff)

        derived = bao_model.derived
        return pars, derived

    def setup_ap(self, config, name):
        self.F_ap[name] = config.getboolean('F_ap')
        self.sig_F_ap[name] = config.getboolean('sig_F_ap')

    def setup_alpha(self, config, name):
        self.alpha[name] = config.getboolean('alpha')
        self.sig_alpha[name] = config.getboolean('sig_alpha')

    def setup_gamma(self, config, name):
        self.gamma[name] = config.getboolean('gamma')
        self.sig_gamma[name] = config.getboolean('sig_gamma')

    def lik_ap(self, dm_dh, name):
        chi2 = (dm_dh - self.F_ap[name])**2 / self.sig_F_ap[name]**2
        return -chi2 / 2

    def lik_alpha(self, dmdh_rdsq, name):
        chi2 = (dmdh_rdsq - self.alpha[name])**2 / self.sig_alpha[name]**2
        return -chi2 / 2

    def lik_gamma(self, dmdh, name):
        chi2 = (dmdh - self.gamma[name])**2 / self.sig_gamma[name]**2
        return -chi2 / 2

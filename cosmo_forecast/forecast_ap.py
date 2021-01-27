import numpy as np
import astropy.constants as const
import astropy.units as units
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import FlatwCDM
from astropy.cosmology import LambdaCDM
from astropy.cosmology import wCDM
from sampler_interface import Sampler


class cosmoModel:
    '''
    BAO model
    Computes position of BAO peak:
    [(D_M(z) / r_d), (H(z) * r_d)]
    '''
    def __init__(self, Omm, H0=None, Omb=None, Ode=None, w0=-1):
        # First setup the parameters we need
        m_nu = np.array([0.02, 0.02, 0.02]) * units.electronvolt
        Neff = 3.046
        Tcmb = 2.7255
        self._Omb = Omb
        self._Omm = Omm

        # Fiducial H_0 - this is needed to initialize astropy.cosmology
        # It has a small impact on the comoving distance through Omega_gamma -
        # this has a 0.05% effect on the (comoving_distance * H0) when going from H0 = 60 to H0 = 80 
        if H0 is None:
            self._H0 = 67.0
        else:
            self._H0 = H0

        # Figure out the cosmology
        if Ode is None:
            # We have flat cosmology (Ode will be inferred from the Friedman eq)
            if w0 == -1:
                self._cosmo = FlatLambdaCDM(H0=self._H0, Om0=Omm, Ob0=Omb, Tcmb0=Tcmb, Neff=Neff, m_nu=m_nu)
            else:
                self._cosmo = FlatwCDM(H0=self._H0, Om0=Omm, w0=w0, Ob0=Omb, Tcmb0=Tcmb, Neff=Neff, m_nu=m_nu)
        else:
            if w0 == -1:
                self._cosmo = LambdaCDM(H0=self._H0, Om0=Omm, Ode0=Ode, Ob0=Omb, Tcmb0=Tcmb, Neff=Neff, m_nu=m_nu)
            else:
                self._cosmo = wCDM(H0=self._H0, Om0=Omm, Ode0=Ode, w0=w0, Ob0=Omb, Tcmb0=Tcmb, Neff=Neff, m_nu=m_nu)

    def compute_ap(self, z_eff):
        '''
        LCDM only for now
        '''
        comoving_dist = self._cosmo.comoving_distance(z=z_eff).value
        c = const.c.to(units.kilometer/units.second).value

        hdm_z = c * comoving_dist * self._H0 * self._cosmo.efunc(z=z_eff)
        return hdm_z

    def compute(self, z_eff, H0_rd):
        '''
        Function for computing the BAO peak coordinates using H0*rd and Omega_m as parameters

        z_eff can be float or 1D array
        '''
        # We need to be careful to take out the H0 dependency in D_m
        # This section of the code is based on astropy
        Ok0 = self._cosmo.Ok0
        cd = self._cosmo.comoving_distance(z=z_eff).value
        c = const.c.to(units.kilometer/units.second).value

        if Ok0 == 0:
            dm_rd = cd * self._H0 / H0_rd
        else:
            sqrtOk0 = np.sqrt(np.abs(Ok0))
            x = cd * self._H0 / c
            if Ok0 < 0:
                dm_rd = np.sin(sqrtOk0 * x) * c / (sqrtOk0 * H0_rd)
            else:
                dm_rd = np.sinh(sqrtOk0 * x) * c / (sqrtOk0 * H0_rd)

        # hrd = H0_rd * self._cosmo.efunc(z=z_eff)
        return dm_rd

class planckLikelihood:
    '''
    Class for computing a likelihood for the
    Planck BAO
    '''
    def __init__(self, config):
        # fid_H0 = config['Model'].getfloat('fid_H0', 67.37)
        # fid_Omb = config['Model'].getfloat('fid_Omega_b', 0.049199)
        # fid_Omm = config['Model'].getfloat('fid_Omega_m', 0.3147)

        self.z_eff = config['Planck'].getfloat('z_eff', 1100)
        self.theta = config['Planck'].getfloat('theta_star_100')
        self.sig_theta = config['Planck'].getfloat('sig_theta_star_100')

        self.omm_hsq = config['Planck'].getfloat('omm_hsq')
        self.sig_omm_hsq = config['Planck'].getfloat('sig_omm_hsq')
        # Compute fiducial model
        # model = cosmoModel(fid_Omm)
        # self.hdm_fid = model.compute_ap(self.z_eff)

    def log_lik(self, theta):
        # print(theta)/
        Omega_m = theta['omega_cdm']
        H0 = theta['h0']
        rd = theta['rd']
        # H0_rd = theta['h0rd']
        H0_rd = H0 * rd
        h = H0/100

        model = cosmoModel(Omega_m)
        dm_rd = model.compute(self.z_eff, H0_rd)
        theory = 1./(dm_rd * 1.02) * 100.
        # print(dm_rd)
        # print(theory)
        log_lik = -0.5 * np.log(2 * np.pi) - np.log(self.sig_theta)
        log_lik -= 0.5 * (theory - self.theta)**2 / self.sig_theta**2

        omm_hsq = H0
        log_lik -= 0.5 * np.log(2 * np.pi) - np.log(self.sig_omm_hsq)
        log_lik -= 0.5 * (omm_hsq - self.omm_hsq)**2 / self.sig_omm_hsq**2
        return log_lik, [H0_rd]


class apForecastLikelihood:
    '''
    Class for computing a forecast likelihood for the
    Alcock-Paczynski parameter
    '''
    def __init__(self, config):
        # fid_H0 = config['Model'].getfloat('fid_H0', 67.37)
        # fid_Omb = config['Model'].getfloat('fid_Omega_b', 0.049199)
        fid_Omm = config['Model'].getfloat('fid_Omega_m', 0.3147)
        self.z_eff = config['Forecast_AP'].getfloat('z_eff', 2.3)
        self.phi_fid = config['Forecast_AP'].getfloat('phi_fid', 1.)
        self.phi_sig = config['Forecast_AP'].getfloat('phi_sig')

        # Compute fiducial model
        model = cosmoModel(fid_Omm)
        self.hdm_fid = model.compute_ap(self.z_eff)

    def log_lik(self, theta):
        Omega_m = theta['omega_m']
        model = cosmoModel(Omega_m)
        theory = model.compute_ap(self.z_eff) / self.hdm_fid

        log_lik = -0.5 * np.log(2 * np.pi) - np.log(self.phi_sig)
        log_lik -= 0.5 * (theory - self.phi_fid)**2 / self.phi_sig**2
        return log_lik


def main(ini_file):
    from configparser import ConfigParser

    # Read ini file and initialize likelihood object
    config = ConfigParser()
    config.read(ini_file)

    limits = {}
    for param, lims in config.items('Likelihood'):
        if param != 'data':
            limits[param] = np.array(lims.split(',')).astype(float)
            assert len(limits[param]) == 2

    # print(limits)
    # lik_obj = apForecastLikelihood(config)
    # print(limits)
    lik_obj = planckLikelihood(config)
    sampler = Sampler(config['Polychord'], limits, lik_obj.log_lik)
    # print(lik_obj.log_lik({'omega_cdm':0.31, 'h0rd':9892}))
    sampler.run()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: ", __file__, " <ini_file>")
    else:
        main(sys.argv[1])

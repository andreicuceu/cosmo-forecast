import numpy as np
import astropy.constants as const
import astropy.units as units
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import FlatwCDM
from astropy.cosmology import LambdaCDM
from astropy.cosmology import wCDM
# from numba import jit
# from . import int_funcs


class baoModel:
    '''
    BAO model
    Computes position of BAO peak:
    [(D_M(z) / r_d), (H(z) * r_d)]
    '''
    def __init__(self, Omm, H0=None, Omb=None, Ode=None, w0=-1):
        # First setup the parameters we need
        m_nu = np.array([0.06, 0.0, 0.0]) * units.electronvolt
        Neff = 3.046
        Tcmb = 2.7255
        self._Omb = Omb
        self._Omm = Omm

        # Fiducial H_0 - this is needed to initialize astropy.cosmology
        # It has a small impact on the comoving distance through Omega_gamma -
        # this has a 0.05% effect on the (comoving_distance * H0) when going from H0 = 60 to H0 = 80 
        if H0 is None:
            self._H0 = 70.0
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

        h = self._H0 / 100.0

        # Check if we need to compute r_d
        if H0 is not None:
            omm = self._Omm * h**2
            omb = self._Omb * h**2
            omnu = self._cosmo.Onu0 * h**2
            self._rd = self.compute_rd_approx(omm, omb, omnu)
        else:
            self._rd = None

        # Container with derived parameters to be read from outside
        self.derived = []
        if Ode is not None:
            self.derived.append(self._cosmo.Ok0)
        if H0 is not None:
            assert(Omb is not None)
            self.derived.append(Omm)
            self.derived.append(self._Omm * h**2)
            self.derived.append(self._Omb * h**2)
            self.derived.append(self._rd)

    # @jit
    def compute_alpha(self, z_eff, H0_rd):
        '''
        Function for computing the isotropic BAO peak = DM * DH / rd^2

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

        hrd = H0_rd * self._cosmo.efunc(z=z_eff)
        dh_rd = c / hrd
        dmdh_rdsq = dm_rd * dh_rd

        return np.sqrt(dmdh_rdsq)

    def compute_ap(self, z_eff):
        '''
        Compute AP parameter = DM/DH
        '''
        Ok0 = self._cosmo.Ok0
        cd = self._cosmo.comoving_distance(z=z_eff).value
        c = const.c.to(units.kilometer/units.second).value

        if Ok0 == 0:
            dm = cd
        else:
            sqrtOk0 = np.sqrt(np.abs(Ok0))
            x = cd * self._H0 / c
            if Ok0 < 0:
                dm = np.sin(sqrtOk0 * x) * c / (sqrtOk0 * self._H0)
            else:
                dm = np.sinh(sqrtOk0 * x) * c / (sqrtOk0 * self._H0)

        H = self._H0 * self._cosmo.efunc(z=z_eff)
        dh = c / H
        dm_dh = dm / dh

        return dm_dh

    def compute_gamma(self, z_eff):
        '''
        Function for computing the isotropic scale gamma = DM * DH
        '''
        # We need to be careful to take out the H0 dependency in D_m
        # This section of the code is based on astropy
        Ok0 = self._cosmo.Ok0
        cd = self._cosmo.comoving_distance(z=z_eff).value
        c = const.c.to(units.kilometer/units.second).value

        if Ok0 == 0:
            dm = cd
        else:
            sqrtOk0 = np.sqrt(np.abs(Ok0))
            x = cd * self._H0 / c
            if Ok0 < 0:
                dm = np.sin(sqrtOk0 * x) * c / (sqrtOk0 * self._H0)
            else:
                dm = np.sinh(sqrtOk0 * x) * c / (sqrtOk0 * self._H0)

        H = self._H0 * self._cosmo.efunc(z=z_eff)
        dh = c / H
        dmdh = dm * dh

        return dmdh

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

        hrd = H0_rd * self._cosmo.efunc(z=z_eff)
        return np.array([dm_rd, hrd])

    # @jit
    def compute_anchored(self, z_eff):
        '''
        Function for computing the BAO peak coordinates using H0, Omega_m and Omega_b as parameters

        z_eff can be float or 1D array
        '''
        assert(self._rd is not None)

        # Compute the BAO peak
        dm_rd = self._cosmo.comoving_transverse_distance(z=z_eff).value / self._rd
        hrd = self._rd * self._H0 * self._cosmo.efunc(z=z_eff)

        return np.array([dm_rd, hrd])

    # @jit
    @staticmethod
    def compute_rd_approx(omm, omb, omnu):
        '''
        Function for analytically computing the sound horizon at the drag epoch
        using eq. 16 from 1411.1074
        '''
        rd = 55.154 * np.exp(-72.3 * (omnu + 0.0006)**2) / (omm**0.25351 * omb**0.12807)
        return rd

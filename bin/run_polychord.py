from cosmo_forecast.likelihoods import baoLikelihood
from cosmo_forecast.sampler_interface import Sampler
import numpy as np
from configparser import ConfigParser


def main(ini_file):

    # Read ini file and initialize likelihood object
    config = ConfigParser()
    config.read(ini_file)

    limits = {}
    for param, lims in config.items('Likelihood'):
        if param != 'data':
            limits[param] = np.array(lims.split(',')).astype(float)
            assert len(limits[param]) == 2

    lik_obj = baoLikelihood(config)
    sampler = Sampler(config['Polychord'], limits, lik_obj.log_lik)
    sampler.run()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: ", __file__, " <ini_file>")
    else:
        main(sys.argv[1])
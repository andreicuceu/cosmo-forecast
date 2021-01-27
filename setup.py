#!/usr/bin/env python
from setuptools import setup, find_packages

requirements = ['numpy', 'scipy', 'astropy', 'numba', 'setuptools']

setup(
    author="Andrei Cuceu",
    author_email='andreicuceu@gmail.com',
    python_requires='>=3.5',
    name='cosmo_forecast',
    packages=find_packages(include=['cosmo_forecast', 'cosmo_forecast.*']),
    install_requires=requirements,
    url='https://github.com/andreicuceu/cosmo_forecast',
    version='0.1.0',
    zip_safe=False,
)

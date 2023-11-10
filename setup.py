from distutils.core import setup
from setuptools import find_packages
import os


# User-friendly description from README.md
current_directory = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except Exception:
    long_description = 'Multivariate Functional Approximations with domain-decomposed B-spline expansions'

setup(
    # Name of the package
    name='mfa-dd-splines',

    # Packages to include into the distribution
    packages=find_packages('.'), 

    # Start with a small number and increase it with every change you make
    # https://semver.org
    version='1.0.0',

    # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    # For example: MIT
    license='BSD',

    # Short description of your library
    description='MFA with Bsplines',

    # Long description of your library
    long_description = long_description,
    long_description_context_type = 'text/markdown',

    # Your name
    author='Vijay Mahadevan', 

    # Your email
    author_email='vijay.m@gmail.com',

    # Either the link to your github or to your website
    url='https://github.com/vijaysm/mfa-dd-splines',

    # Link from which the project can be downloaded
    download_url='https://github.com/vijaysm/mfa-dd-splines',

    # List of keyword arguments
    keywords=['MFA', 'splines', 'domain decomposition'],

    # List of packages to install with this one
    install_requires=[
        'autograd==1.6.2', 
        'matplotlib==3.8.1',
        'mpi4py==3.1.5',
        'numpy==1.26.1',
        'packaging==23.2',
        'pyvista==0.42.3',
        'scipy==1.11.3',
        'Splipy==1.7.4',
        'uvw==0.5.2'],

    # https://pypi.org/classifiers/
    classifiers=[]  
)


'''
Created on 12 April 2017
@author: Sambit Giri
Setup script
'''

from setuptools import setup, find_packages
#from distutils.core import setup


setup(name='psi',
      version='0.0.1',
      author='Sambit Giri',
      author_email='sambit.giri@gmail.com',
      package_dir = {'psi' : 'src'},
      packages=['psi'],
      package_data={'share':['*'],},
      install_requires=['numpy','scipy','scikit-learn','scikit-image', 'pyDOE', 'astropy', 'pyfits', 'pyfftw', 'getdist'],
      #include_package_data=True,
)

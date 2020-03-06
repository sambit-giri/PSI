import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from importlib import reload
import psi

yerr_param = [0.0, 2.5]#[0.1, 0.5]
line  = psi.sample_models.noisy_line(yerr_param=yerr_param)
xs    = line.xs()
y_obs = line.observation()

#### For BOLFI
simulator = lambda x: line.simulator(x, line.true_intercept)
distance  = psi.distances.euclidean

#### BOLFI
#from pyDOE import *
#lhd = lhs(2, samples=5)

# 1 param
prior  = {'m': 'uniform'}#, 'c': 'uniform'}
bounds = {'m': [-2.5, 0.5]}#, 'c': [0,10]}
gpr = GaussianProcessRegressor()

rn = psi.BOLFI_1param(simulator, distance, y_obs, prior, bounds, N_init=5, gpr=gpr)

# 2 param
simulator = lambda x: line.simulator(x[0], x[1])
distance  = psi.distances.euclidean

prior  = {'m': 'uniform', 'c': 'uniform'}
bounds = {'m': [-2.5, 0.5], 'c': [0,10]}
gpr = GaussianProcessRegressor()

rn = psi.BOLFI(simulator, distance, y_obs, prior, bounds, N_init=5, gpr=gpr)

	

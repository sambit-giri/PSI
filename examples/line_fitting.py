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
rn.run()

#Plot
plt.subplot(121)
plt.plot(rn.xout, rn.post_mean_normmax[25])
#plt.plot(rn.successive_JS_dist, c='C0')
plt.subplot(122)
plt.plot(rn.cv_JS_dist['mean'], c='C1')	

# 2 param
simulator = lambda x: line.simulator(x[0], x[1])
distance  = psi.distances.euclidean

prior  = {'m': 'uniform', 'c': 'uniform'}
bounds = {'m': [-2.5, 0.5], 'c': [0,10]}
gpr = GaussianProcessRegressor()

rn = psi.BOLFI(simulator, distance, y_obs, prior, bounds, N_init=5, gpr=gpr, successive_JS_tol=0.02)
rn.run()	
	
## JS over iterations
plt.rcParams['figure.figsize'] = [12, 6]

plt.subplot(121)
#plt.plot(rn.xout, rn.post_mean_normmax[1])
plt.plot(rn.successive_JS_dist, c='C0')
plt.subplot(122)
plt.plot(rn.cv_JS_dist['mean'], c='C1')	

# Plot
plt.rcParams['figure.figsize'] = [12, 6]
plt.subplot(121)
plt.title('Distances')
plt.scatter(rn.params[:,0], rn.params[:,1], c=rn.dists, cmap='jet')
plt.colorbar()
plt.subplot(122)
plt.title('Posterior')
plt.scatter(rn.xout[:,0], rn.xout[:,1], c=rn.post_mean_normmax[-1].flatten(), cmap='jet')
plt.colorbar()
plt.scatter(line.true_slope,line.true_intercept, marker='*', c='k')


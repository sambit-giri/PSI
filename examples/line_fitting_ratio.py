import numpy as np
from importlib import reload
import psi

yerr_param = [0.0, 1.0]#[0.1, 1.5]
line  = psi.sample_models.noisy_line(yerr_param=yerr_param)
xs    = line.xs()
y_obs = line.observation()

## LFIRE
# 1 param
simulator = lambda x: line.simulator(x, line.true_intercept)
distance  = psi.distances.euclidean

prior  = {'m': 'uniform'}#, 'c': 'uniform'}
bounds = {'m': [-2.5, 0.5]}#, 'c': [0,10]}

lfi = LFIRE(simulator, distance, y_obs, prior, bounds, sim_out_den=None, n_m=100, n_theta=100, n_grid_out=100, thetas=None)
lfi.run()

plt.plot(lfi.thetas, lfi.posterior)

# 2 param
simulator2 = lambda x: line.simulator(x[0], x[1])
distance2  = psi.distances.euclidean

prior2  = {'m': 'uniform', 'c': 'uniform'}
bounds2 = {'m': [-2.5, 0.5], 'c': [0,10]}

lfi2 = LFIRE(simulator2, distance2, y_obs, prior2, bounds2, sim_out_den=None, n_m=100, n_theta=100, n_grid_out=10, thetas=None)
lfi2.run()

plt.scatter(lfi2.thetas[:,0], lfi2.thetas[:,0], c=lfi2.posterior)
plt.colorbar()



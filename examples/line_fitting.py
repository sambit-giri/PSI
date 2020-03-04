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
from pyDOE import *
lhd = lhs(2, samples=5)

prior  = {'m': 'uniform'}#, 'c': 'uniform'}
bounds = {'m': [-2.5, 0.5]}#, 'c': [0,10]}
gpr = GaussianProcessRegressor()

rn = psi.BOLFI_1param(simulator, distance, y_obs, prior, bounds, N_init=5, gpr=gpr)

# Start
bounds = np.array([[-2.5, 0.5], [0,10]])
prior_slope     = lambda: numpy.random.uniform(low=-2.5, high=0.5, size=1)
prior_intercept = lambda: numpy.random.uniform(low=0, high=10, size=1)

N_init  = 5
params  = np.array([prior_slope() for i in range(N_init)]).squeeze()
sim_out = np.array([simulator(i) for i in params])
dists   = np.array([distance(y_obs, ss) for ss in sim_out])

X = params.reshape(-1,1) if params.ndim==1 else params
y = dists.reshape(-1,1) if dists.ndim==1 else dists

#kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor()#kernel=kernel, random_state=0)
gpr.fit(X, y)
gpr.score(X, y)

xplot = np.sort(np.array([prior_slope() for i in range(100)]), axis=0)
y_pred, y_std = gpr.predict(xplot, return_std=True)
unnorm_post_mean = np.exp(-y_pred/2.)
#norm = simps(unnorm_post_mean.squeeze(), xplot.squeeze())
#norm_post_mean = unnorm_post_mean/norm

## Next step
# Obtain next sampling point from the acquisition function (expected_improvement)
X_next = psi.bayesian_optimisation.propose_location(psi.bayesian_optimisation.expected_improvement, X, y, gpr, bounds[0].reshape(1,-1))
    
# Obtain next noisy sample from the objective function
y_next = simulator(X_next)
d_next = distance(y_obs, y_next)

params = np.append(params, X_next) 
dists  = np.append(dists, d_next) 

X = params.reshape(-1,1) if params.ndim==1 else params
y = dists.reshape(-1,1) if dists.ndim==1 else dists

#kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor()#kernel=kernel, random_state=0)
gpr.fit(X, y)
gpr.score(X, y)

xplot = np.sort(np.array([prior_slope() for i in range(100)]), axis=0)
y_pred, y_std = gpr.predict(xplot, return_std=True)
unnorm_post_mean = np.exp(-y_pred/2.)


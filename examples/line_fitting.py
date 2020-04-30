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
	
	
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.model_selection import KFold
from scipy.integrate import simps
from psi import distances
from psi import bayesian_optimisation as bopt
from psi import helper_functions as hf 

def grid_bounds(bounds, n_grid=20):
	def add_dim_to_grid(bound, n_grid=100, init_grid=None):
		if init_grid is None:
			final_grid = np.linspace(bound[0], bound[1], n_grid)
		else:
			final_grid = [np.append(gg,nn) if type(gg)==np.ndarray else [gg,nn] for gg in init_grid for nn in np.linspace(bound[0], bound[1], n_grid)]
		return np.array(final_grid)
	ndim_param = bounds.shape[0]
	grid = add_dim_to_grid(bounds[0], n_grid=n_grid, init_grid=None)
	if ndim_param==1: return grid
	for bound in bounds[1:]:
		grid = add_dim_to_grid(bound, n_grid=n_grid, init_grid=grid)
	return grid

class BOLFI:
	def __init__(self, simulator, distance, observation, prior, bounds, N_init=5, gpr=None, max_iter=100, cv_JS_tol=0.01, successive_JS_tol=0.01, n_grid_out=100):
		self.N_init  = N_init
		self.gpr = GaussianProcessRegressor() if gpr is None else gpr
		self.simulator = simulator
		self.distance  = distance
		self.y_obs = observation
		self.param_names = [kk for kk in prior]
		self.param_bound = bounds
		self.bounds = np.array([bounds[kk] for kk in self.param_names])
		#self.sample_prior = {}
		#for i,kk in enumerate(self.param_names):
		#	self.sample_prior[kk] = lambda: bounds[kk][0]+(bounds[kk][1]-bounds[kk][0])*np.random.uniform()
		self.xout = grid_bounds(self.bounds, n_grid=n_grid_out)
		self.max_iter = max_iter
		self.params = np.array([])
		self.post_mean_unnorm  = []
		self.post_mean_normmax = []
		self.cv_JS_tol  = cv_JS_tol
		self.cv_JS_dist = {'mean':[], 'std':[]}
		self.successive_JS_tol  = successive_JS_tol
		self.successive_JS_dist = []		

	def sample_prior(self, kk):
		return self.param_bound[kk][0]+(self.param_bound[kk][1]-self.param_bound[kk][0])*np.random.uniform()

	def fit_model(self, params, dists):
		X = params.reshape(-1,1) if params.ndim==1 else params
		y = dists.reshape(-1,1) if dists.ndim==1 else dists

		n_cv = 10 if y.size>20 else 5
		kf = KFold(n_splits=n_cv)
		pdfs = []
		for train_index, test_index in kf.split(X):
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			self.gpr.fit(X_train, y_train)
			y_pred, y_std = self.gpr.predict(self.xout, return_std=True)
			unnorm_post_mean = np.exp(-y_pred/2.)
			pdfs.append(unnorm_post_mean)
		
		cvdist = np.array([distances.jensenshannon(p1,p2) for p1 in pdfs for p2 in pdfs])
		self.cv_JS_dist['std'].append(cvdist.std())
		self.cv_JS_dist['mean'].append(cvdist.mean())
		y_pred, y_std = self.gpr.predict(self.xout, return_std=True)
		unnorm_post_mean = np.exp(-y_pred/2.)
		self.post_mean_unnorm.append(unnorm_post_mean)
		self.post_mean_normmax.append(unnorm_post_mean/unnorm_post_mean.max())
		return cvdist.std()

	def run(self, max_iter=None):
		if max_iter is not None: self.max_iter = max_iter
		#gpr = self.gpr
		start_iter = self.params.size
		# Initialization
		if start_iter<self.N_init:
			params  = np.array([[self.sample_prior(kk) for kk in self.param_names] for i in range(self.N_init)]).squeeze()
			sim_out = np.array([self.simulator(i) for i in params])
			dists   = np.array([self.distance(self.y_obs, ss) for ss in sim_out])
			self.params = params
			self.dists  = dists
			msg = self.fit_model(self.params, self.dists)
			hf.loading_verbose('{0:.6f}'.format(msg))
		
		# Further sampling
		start_iter = len(self.params)
		condition1, condition2 = False, False
		for n_iter in range(start_iter,self.max_iter):
			if condition1 and condition2: break
			X = self.params.reshape(-1,1) if self.params.ndim==1 else self.params
			y = self.dists.reshape(-1,1) if self.dists.ndim==1 else self.dists
			X_next = bopt.propose_location(bopt.expected_improvement, X, y, self.gpr, self.bounds).T

			y_next = self.simulator(X_next.T)
			d_next = self.distance(self.y_obs, y_next)

			self.params = np.vstack((self.params, X_next)) #np.append(self.params, X_next)
			self.dists  = np.append(self.dists, d_next)
			msg = self.fit_model(self.params, self.dists)
			hf.loading_verbose('{0:6d}|{1:.6f}'.format(n_iter+1,msg))
			sucJSdist   = distances.jensenshannon(self.post_mean_normmax[-1], self.post_mean_normmax[-2])[0]
			self.successive_JS_dist.append(sucJSdist)
			condition1 = self.cv_JS_dist['mean'][-1]+self.cv_JS_dist['std'][-1]<self.cv_JS_tol
			condition2 = self.successive_JS_dist[-1]<self.successive_JS_tol
		


	

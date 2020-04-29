import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.model_selection import KFold
from scipy.integrate import simps
from . import distances
from . import bayesian_optimisation as bopt
from . import helper_functions as hf 

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

class BOLFI_1param:
	def __init__(self, simulator, distance, observation, prior, bounds, N_init=5, gpr=None, max_iter=100, cv_JS_tol=0.01, successive_JS_tol=0.01):
		self.N_init  = N_init
		self.gpr = GaussianProcessRegressor() if gpr is None else gpr
		self.simulator = simulator
		self.distance  = distance
		self.y_obs = observation
		param_names = [kk for kk in prior]
		self.bounds = np.array([bounds[kk] for kk in param_names])
		self.sample_prior = lambda: np.random.uniform(low=self.bounds[0,0], high=self.bounds[0,1], size=1)
		self.xout = np.sort(np.array([self.sample_prior() for i in range(100)]), axis=0)
		self.max_iter = max_iter
		self.params = np.array([])
		self.post_mean_unnorm  = []
		self.post_mean_normmax = []
		self.cv_JS_tol  = cv_JS_tol
		self.cv_JS_dist = {'mean':[], 'std':[]}
		self.successive_JS_tol  = successive_JS_tol
		self.successive_JS_dist = []

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
			y_pred, y_std = gpr.predict(self.xout, return_std=True)
			unnorm_post_mean = np.exp(-y_pred/2.)
			pdfs.append(unnorm_post_mean)
		
		cvdist = np.array([distances.jensenshannon(p1,p2) for p1 in pdfs for p2 in pdfs])
		self.cv_JS_dist['std'].append(cvdist.std())
		self.cv_JS_dist['mean'].append(cvdist.mean())
		y_pred, y_std = gpr.predict(self.xout, return_std=True)
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
			params  = np.array([self.sample_prior() for i in range(self.N_init)]).squeeze()
			sim_out = np.array([self.simulator(i) for i in params])
			dists   = np.array([self.distance(self.y_obs, ss) for ss in sim_out])
			self.params = params
			self.dists  = dists
			msg = self.fit_model(self.params, self.dists)
			hf.loading_verbose(str(msg))
		# Further sampling
		start_iter = self.params.size
		condition1, condition2 = False, False
		for n_iter in range(start_iter,self.max_iter):
			if condition1 and condition2: break
			X = self.params.reshape(-1,1) if self.params.ndim==1 else self.params
			y = self.dists.reshape(-1,1) if self.dists.ndim==1 else self.dists
			X_next = bopt.propose_location(bopt.expected_improvement, X, y, self.gpr, self.bounds[0].reshape(1,-1))

			y_next = self.simulator(X_next)
			d_next = self.distance(self.y_obs, y_next)

			self.params = np.append(self.params, X_next) 
			self.dists  = np.append(self.dists, d_next)
			msg = self.fit_model(self.params, self.dists)
			hf.loading_verbose(str(msg))
			sucJSdist   = distances.jensenshannon(self.post_mean_normmax[-1], self.post_mean_normmax[-2])[0]
			self.successive_JS_dist.append(sucJSdist)
			condition1 = self.cv_JS_dist['mean'][-1]+self.cv_JS_dist['std'][-1]<self.cv_JS_tol
			condition2 = self.successive_JS_dist[-1]<self.successive_JS_tol

class BOLFI:
	def __init__(self, simulator, distance, observation, prior, bounds, N_init=5, gpr=None, max_iter=100, cv_JS_tol=0.01, successive_JS_tol=0.01):
		self.N_init  = N_init
		self.gpr = GaussianProcessRegressor() if gpr is None else gpr
		self.simulator = simulator
		self.distance  = distance
		self.y_obs = observation
		self.param_names = [kk for kk in prior]
		self.bounds = np.array([bounds[kk] for kk in self.param_names])
		self.sample_prior = {}
		for i,kk in enumerate(self.param_names):
			self.sample_prior[kk] = lambda: np.random.uniform(low=self.bounds[i,0], high=self.bounds[i,1], size=1)
		self.xout = grid_bounds(self.bounds, n_grid=20)
		self.max_iter = max_iter
		self.params = np.array([])
		self.post_mean_unnorm  = []
		self.post_mean_normmax = []
		self.cv_JS_tol  = cv_JS_tol
		self.cv_JS_dist = {'mean':[], 'std':[]}
		self.successive_JS_tol  = successive_JS_tol
		self.successive_JS_dist = []		
	def run(self, max_iter=None):
		if max_iter is not None: self.max_iter = max_iter
		gpr = self.gpr
		start_iter = self.params.size
		def fit_model(params, dists):
			X = params.reshape(-1,1) if params.ndim==1 else params
			y = dists.reshape(-1,1) if dists.ndim==1 else dists

			n_cv = 10 if y.size>20 else 5
			kf = KFold(n_splits=n_cv)
			pdfs = []
			for train_index, test_index in kf.split(X):
				X_train, X_test = X[train_index], X[test_index]
				y_train, y_test = y[train_index], y[test_index]
				gpr.fit(X_train, y_train)
				y_pred, y_std = gpr.predict(self.xout, return_std=True)
				unnorm_post_mean = np.exp(-y_pred/2.)
				pdfs.append(unnorm_post_mean)
		
			cvdist = np.array([distances.jensenshannon(p1,p2) for p1 in pdfs for p2 in pdfs])
			self.cv_JS_dist['std'].append(cvdist.std())
			self.cv_JS_dist['mean'].append(cvdist.mean())
			y_pred, y_std = gpr.predict(self.xout, return_std=True)
			unnorm_post_mean = np.exp(-y_pred/2.)
			self.post_mean_unnorm.append(unnorm_post_mean)
			self.post_mean_normmax.append(unnorm_post_mean/unnorm_post_mean.max())
			return cvdist.std()
		# Initialization
		if start_iter<self.N_init:
			params  = np.array([[self.sample_prior[kk]()[0] for kk in self.param_names] for i in range(self.N_init)]).squeeze()
			sim_out = np.array([self.simulator(i) for i in params])
			dists   = np.array([self.distance(self.y_obs, ss) for ss in sim_out])
			self.params = params
			self.dists  = dists
			msg = fit_model(self.params, self.dists)
			hf.loading_verbose(str(msg))
		
		# Further sampling
		start_iter = len(self.params)
		condition1, condition2 = False, False
		for n_iter in range(start_iter,self.max_iter):
			if condition1 and condition2: break
			X = self.params.reshape(-1,1) if self.params.ndim==1 else self.params
			y = self.dists.reshape(-1,1) if self.dists.ndim==1 else self.dists
			X_next = bopt.propose_location(bopt.expected_improvement, X, y, gpr, self.bounds).T

			y_next = self.simulator(X_next.T)
			d_next = self.distance(self.y_obs, y_next)

			self.params = np.vstack((self.params, X_next)) #np.append(self.params, X_next)
			self.dists  = np.append(self.dists, d_next)
			msg = fit_model(self.params, self.dists)
			hf.loading_verbose(str(msg))
			sucJSdist   = distances.jensenshannon(self.post_mean_normmax[-1], self.post_mean_normmax[-2])[0]
			self.successive_JS_dist.append(sucJSdist)
			condition1 = self.cv_JS_dist['mean'][-1]+self.cv_JS_dist['std'][-1]<self.cv_JS_tol
			condition2 = self.successive_JS_dist[-1]<self.successive_JS_tol


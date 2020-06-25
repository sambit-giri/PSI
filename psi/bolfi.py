import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.model_selection import KFold
from scipy.integrate import simps
import warnings 
warnings.filterwarnings("ignore")
from . import distances
from . import bayesian_optimisation as bopt
from . import helper_functions as hf 

def _grid_bounds(bounds, n_grid=20):
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
	def __init__(self, simulator, distance, observation, prior, bounds, N_init=5, gpr=None, max_iter=100, cv_JS_tol=0.01, successive_JS_tol=0.01, exploitation_exploration=None, sigma_tol=0.001):
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

		self.exploitation_exploration = exploitation_exploration
		self.sigma_tol = sigma_tol

	def fit_model(self, params, dists, cv=True):
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
		norm_post_mean   = unnorm_post_mean/unnorm_post_mean.max()
		norm_post_std    = 0.5*y_std.flatten()*norm_post_mean.flatten()
		self.sigma_theta = norm_post_std
		self.post_mean_unnorm.append(unnorm_post_mean)
		self.post_mean_normmax.append(norm_post_mean)
		return self.cv_JS_dist['mean'][-1] #cvdist.std()

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
			hf.loading_verbose('{0:.6f}'.format(msg))
		# Further sampling
		start_iter = self.params.size
		condition1, condition2 = False, False
		for n_iter in range(start_iter,self.max_iter):
			if condition1 and condition2: break
			X = self.params.reshape(-1,1) if self.params.ndim==1 else self.params
			y = self.dists.reshape(-1,1) if self.dists.ndim==1 else self.dists
			
			if self.sigma_tol is not None:
				self.exploitation_exploration = 1./self.sigma_tol if np.any(self.sigma_theta>self.sigma_tol) else 1.
			#X_next = bopt.propose_location(bopt.expected_improvement, X, y, self.gpr, self.bounds[0].reshape(1,-1), n_restarts=10).T
			X_next = bopt.propose_location(bopt.negativeGP_LCB, X, y, self.gpr, self.bounds[0].reshape(1,-1), n_restarts=10, xi=self.exploitation_exploration).T

			y_next = self.simulator(X_next)
			d_next = self.distance(self.y_obs, y_next)

			self.params = np.append(self.params, X_next) 
			self.dists  = np.append(self.dists, d_next)
			sucJSdist   = distances.jensenshannon(self.post_mean_normmax[-1], self.post_mean_normmax[-2])[0] if len(self.post_mean_normmax)>1 else 10
			self.successive_JS_dist.append(sucJSdist)

			msg = self.fit_model(self.params, self.dists)
			hf.loading_verbose('{0:6d}|{1:.6f}|{1:.6f}'.format(n_iter+1,msg,sucJSdist))
			#condition1 = self.cv_JS_dist['mean'][-1]+self.cv_JS_dist['std'][-1]<self.cv_JS_tol
			condition1 = self.cv_JS_dist['mean'][-1]<self.cv_JS_tol
			condition2 = self.successive_JS_dist[-1]<self.successive_JS_tol

class BOLFI:
	def __init__(self, simulator, distance, observation, prior, bounds, N_init=5, gpr=None, max_iter=100, cv_JS_tol=0.01, successive_JS_tol=0.01, n_grid_out=100, exploitation_exploration=None, sigma_tol=0.001, inside_nSphere=False, fill_value=np.nan, params=None, dists=None, batch=1):
		self.N_init  = N_init
		self.gpr = GaussianProcessRegressor() if gpr is None else gpr
		self.simulator = simulator
		self.distance  = distance
		self.y_obs = observation
		self.param_names = [kk for kk in prior]
		self.param_bound = bounds
		self.bounds = np.array([bounds[kk] for kk in self.param_names])
		self.bound_mins = self.bounds.min(axis=1)
		self.bound_maxs = self.bounds.max(axis=1)
		#self.sample_prior = {}
		#for i,kk in enumerate(self.param_names):
		#	self.sample_prior[kk] = lambda: bounds[kk][0]+(bounds[kk][1]-bounds[kk][0])*np.random.uniform()
		self.batch = batch

		self.xout = _grid_bounds(self.bounds, n_grid=n_grid_out)
		self.max_iter = max_iter
		self.params = np.array([])
		self.post_mean_unnorm  = []
		self.post_mean_normmax = []
		self.cv_JS_tol  = cv_JS_tol
		self.cv_JS_dist = {'mean':[], 'std':[]}
		self.successive_JS_tol  = successive_JS_tol
		self.successive_JS_dist = []	

		self.exploitation_exploration = exploitation_exploration
		self.sigma_tol = sigma_tol	
		self.inside_nSphere = inside_nSphere
		self.fill_value = fill_value

		if params is not None and dists is None:
			dists = np.array([self.sim_n_dist(i) for i in params])

		self.params = np.array([]) if params is None else params
		self.dists  = np.array([]) if dists is None else dists

	def sample_prior(self, kk):
		return self.param_bound[kk][0]+(self.param_bound[kk][1]-self.param_bound[kk][0])*np.random.uniform()

	def sim_n_dist(self, xi):
		if self.inside_nSphere:
			xr = np.sum(((xi-self.bound_mins)/(self.bound_maxs-self.bound_mins)-0.5)**2)
			if xr>0.25: return self.fill_value
		yi = self.simulator(xi)
		di = self.distance(self.y_obs, yi)
		return di

	def fit_model(self, params, dists):
		X = params.reshape(-1,1) if params.ndim==1 else params
		y = dists.reshape(-1,1) if dists.ndim==1 else dists

		n_cv = 10 if y.size>50 else 5
		kf = KFold(n_splits=n_cv)
		pdfs = []
		for train_index, test_index in kf.split(X):
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			args = np.isfinite(y_train.flatten()) 
			self.gpr.fit(X_train[args], y_train[args])
			y_pred, y_std = self.gpr.predict(self.xout, return_std=True)
			unnorm_post_mean = np.exp(-y_pred/2.)
			pdfs.append(unnorm_post_mean)
		
		cvdist = np.array([distances.jensenshannon(p1,p2) for p1 in pdfs for p2 in pdfs])
		self.cv_JS_dist['std'].append(cvdist.std())
		self.cv_JS_dist['mean'].append(cvdist.mean())

		args = np.isfinite(y.flatten()) 
		self.gpr.fit(X[args], y[args])
		y_pred, y_std = self.gpr.predict(self.xout, return_std=True)
		unnorm_post_mean = np.exp(-y_pred/2.)
		norm_post_mean   = unnorm_post_mean/unnorm_post_mean.max()
		norm_post_std    = 0.5*y_std.flatten()*norm_post_mean.flatten()
		self.sigma_theta = norm_post_std
		self.post_mean_unnorm.append(unnorm_post_mean)
		self.post_mean_normmax.append(norm_post_mean)
		return self.cv_JS_dist['mean'][-1] #cvdist.std()

	def get_next_point(self):
		X = self.params.reshape(-1,1) if self.params.ndim==1 else self.params
		y = self.dists.reshape(-1,1) if self.dists.ndim==1 else self.dists

		if self.sigma_tol is not None:
			self.exploitation_exploration = 1./self.sigma_tol if np.any(self.sigma_theta>self.sigma_tol) else 1.
		args = np.isfinite(y.flatten())
		#X_next = bopt.propose_location(bopt.expected_improvement, self._adjust_shape(self.params), self.posterior_params, self.gpr, self.lfi.bounds, n_restarts=10).T
		X_next = bopt.propose_location_nSphere(bopt.negativeGP_LCB, X[args,:], y[args,:], self.gpr, self.bounds, n_restarts=10, xi=self.exploitation_exploration, batch=self.batch).T
		return X_next

	def run(self, max_iter=None, trained_gpr=True):
		if max_iter is not None: self.max_iter = max_iter
		#gpr = self.gpr
		start_iter = len(self.params)
		# Initialization
		if start_iter<self.N_init:
			params  = np.array([[self.sample_prior(kk) for kk in self.param_names] for i in range(self.N_init)]).squeeze()
			#sim_out = np.array([self._simulator(i) for i in params])
			#dists   = np.array([self.distance(self.y_obs, ss) for ss in sim_out])
			dists = np.array([self.sim_n_dist(i) for i in params])
			self.params = params
			self.dists  = dists
			
		msg = self.fit_model(self.params, self.dists)
		hf.loading_verbose('{0:.6f}'.format(msg))
		
		# Further sampling
		start_iter = len(self.params)
		condition1, condition2 = False, False
		#for start_iter in range(start_iter,self.max_iter):
		while start_iter<self.max_iter:
			if condition1 and condition2: break
			
			X_next = self.get_next_point()
			d_next = np.array([self.sim_n_dist(X_n.T) for X_n in X_next])

			self.params = np.append(self.params, X_next, axis=0) 
			self.dists  = np.append(self.dists, d_next)
			sucJSdist   = distances.jensenshannon(self.post_mean_normmax[-1], self.post_mean_normmax[-2])[0] if len(self.post_mean_normmax)>1 else 10
			self.successive_JS_dist.append(sucJSdist)

			start_iter = len(self.params)
			msg = self.fit_model(self.params, self.dists)
			hf.loading_verbose('{0:6d}|{1:.6f}|{1:.6f}'.format(start_iter+1,msg,sucJSdist))
			#condition1 = self.cv_JS_dist['mean'][-1]+self.cv_JS_dist['std'][-1]<self.cv_JS_tol
			condition1 = self.cv_JS_dist['mean'][-1]<self.cv_JS_tol
			condition2 = self.successive_JS_dist[-1]<self.successive_JS_tol

		if trained_gpr:
			print('\nFinal training of GPR for output.')
			X = self.params.reshape(-1,1) if self.params.ndim==1 else self.params
			y = self.dists.reshape(-1,1) if self.dists.ndim==1 else self.dists
			args = np.isfinite(y.flatten()) 
			self.gpr.fit(X[args], y[args])


class BOLFI_postGPR:
	def __init__(self, simulator, distance, observation, prior, bounds, N_init=5, gpr=None, max_iter=100, cv_JS_tol=0.01, successive_JS_tol=0.01, n_grid_out=100, exploitation_exploration=None, sigma_tol=0.001, inside_nSphere=True, fill_value=1000):
		self.N_init  = N_init
		self.gpr = GaussianProcessRegressor() if gpr is None else gpr
		self.simulator = simulator
		self.distance  = distance
		self.y_obs = observation
		self.param_names = [kk for kk in prior]
		self.param_bound = bounds
		self.bounds = np.array([bounds[kk] for kk in self.param_names])
		self.bound_mins = self.bounds.min(axis=1)
		self.bound_maxs = self.bounds.max(axis=1)
		#self.sample_prior = {}
		#for i,kk in enumerate(self.param_names):
		#	self.sample_prior[kk] = lambda: bounds[kk][0]+(bounds[kk][1]-bounds[kk][0])*np.random.uniform()
		self.xout = _grid_bounds(self.bounds, n_grid=n_grid_out)
		self.max_iter = max_iter
		self.params = np.array([])
		self.post_mean_unnorm  = []
		self.post_mean_normmax = []
		self.cv_JS_tol  = cv_JS_tol
		self.cv_JS_dist = {'mean':[], 'std':[]}
		self.successive_JS_tol  = successive_JS_tol
		self.successive_JS_dist = []	

		self.exploitation_exploration = exploitation_exploration
		self.sigma_tol = sigma_tol	
		self.inside_nSphere = inside_nSphere
		self.fill_value = fill_value

	def sample_prior(self, kk):
		return self.param_bound[kk][0]+(self.param_bound[kk][1]-self.param_bound[kk][0])*np.random.uniform()

	def sim_n_dist(self, xi):
		if self.inside_nSphere:
			xr = np.sum(((xi-self.bound_mins)/(self.bound_maxs-self.bound_mins)-0.5)**2)
			if xr>0.25: return self.fill_value
		yi = self.simulator(xi)
		di = self.distance(self.y_obs, yi)
		return di

	def fit_model(self, params, dists):
		X = params.reshape(-1,1) if params.ndim==1 else params
		y = dists.reshape(-1,1) if dists.ndim==1 else dists

		n_cv = 10 if y.size>50 else 5
		kf = KFold(n_splits=n_cv)
		pdfs = []
		for train_index, test_index in kf.split(X):
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			self.gpr.fit(X_train, np.exp(-y_train/2.))
			y_pred, y_std = self.gpr.predict(self.xout, return_std=True)
			unnorm_post_mean = y_pred
			pdfs.append(unnorm_post_mean)
		
		cvdist = np.array([distances.jensenshannon(p1,p2) for p1 in pdfs for p2 in pdfs])
		self.cv_JS_dist['std'].append(cvdist.std())
		self.cv_JS_dist['mean'].append(cvdist.mean())

		self.gpr.fit(X, np.exp(-y/2.))
		y_pred, y_std = self.gpr.predict(self.xout, return_std=True)
		unnorm_post_mean = y_pred
		norm_post_mean   = unnorm_post_mean/unnorm_post_mean.max()
		norm_post_std    = y_std.flatten()
		self.sigma_theta = norm_post_std
		self.post_mean_unnorm.append(unnorm_post_mean)
		self.post_mean_normmax.append(norm_post_mean)
		return self.cv_JS_dist['mean'][-1] #cvdist.std()

	def run(self, max_iter=None, trained_gpr=True):
		if max_iter is not None: self.max_iter = max_iter
		#gpr = self.gpr
		start_iter = self.params.size
		# Initialization
		if start_iter<self.N_init:
			params  = np.array([[self.sample_prior(kk) for kk in self.param_names] for i in range(self.N_init)]).squeeze()
			#sim_out = np.array([self._simulator(i) for i in params])
			#dists   = np.array([self.distance(self.y_obs, ss) for ss in sim_out])
			dists = np.array([self.sim_n_dist(i) for i in params])
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

			if self.sigma_tol is not None:
				self.exploitation_exploration = 1./self.sigma_tol if np.any(self.sigma_theta>self.sigma_tol) else 1.
			#X_next = bopt.propose_location(bopt.expected_improvement, self._adjust_shape(self.params), self.posterior_params, self.gpr, self.lfi.bounds, n_restarts=10).T
			X_next = bopt.propose_location(bopt.negativeGP_LCB, X, np.exp(-y/2.), self.gpr, self.bounds, n_restarts=10, xi=self.exploitation_exploration).T

			#y_next = self._simulator(X_next.T)
			#d_next = self.distance(self.y_obs, y_next)
			d_next = self.sim_n_dist(X_next.T)

			self.params = np.append(self.params, X_next, axis=0) 
			self.dists  = np.append(self.dists, d_next)
			sucJSdist   = distances.jensenshannon(self.post_mean_normmax[-1], self.post_mean_normmax[-2])[0] if len(self.post_mean_normmax)>1 else 10
			self.successive_JS_dist.append(sucJSdist)

			msg = self.fit_model(self.params, self.dists)
			hf.loading_verbose('{0:6d}|{1:.6f}|{1:.6f}'.format(n_iter+1,msg,sucJSdist))
			#condition1 = self.cv_JS_dist['mean'][-1]+self.cv_JS_dist['std'][-1]<self.cv_JS_tol
			condition1 = self.cv_JS_dist['mean'][-1]<self.cv_JS_tol
			condition2 = self.successive_JS_dist[-1]<self.successive_JS_tol

		if trained_gpr:
			print('\nFinal training of GPR for output.')
			X = self.params.reshape(-1,1) if self.params.ndim==1 else self.params
			y = self.dists.reshape(-1,1) if self.dists.ndim==1 else self.dists
			self.gpr.fit(X, np.exp(-y/2.))


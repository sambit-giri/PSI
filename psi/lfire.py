import numpy as np
from sklearn.model_selection import KFold
from scipy.integrate import simps
import warnings 
warnings.filterwarnings("ignore")
from . import distances
from . import helper_functions as hf
from sklearn.linear_model import LogisticRegressionCV

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


class LFIRE_core:
	def __init__(self, simulator, observation, prior, bounds, sim_out_den=None, n_m=100, n_theta=100, n_grid_out=100, thetas=None, verbose=True, penalty='l1', n_jobs=4, clfy=None):
		#self.N_init  = N_init
		self.simulator = simulator
		#self.distance  = distance
		self.verbose = verbose
		self.penalty = penalty
		self.y_obs = observation
		self.param_names = [kk for kk in prior]
		self.param_bound = bounds
		self.bounds = np.array([bounds[kk] for kk in self.param_names])
		self.n_m = n_m
		self.n_theta = n_theta
		self.n_grid_out = n_grid_out
		self.n_jobs = n_jobs
		self.clfy   = clfy

		if sim_out_den is not None: 
			self.sim_out_den = sim_out_den
			self.n_theta = sim_out_den.shape[0]
		else: 
			self.sim_denominator()  

		if thetas is None: self.theta_grid()
		else: self.thetas = thetas 

	def sample_prior(self, kk):
		return self.param_bound[kk][0]+(self.param_bound[kk][1]-self.param_bound[kk][0])*np.random.uniform()

	def theta_grid(self):
		self.thetas = _grid_bounds(self.bounds, n_grid=self.n_grid_out)

	def sim_denominator(self):
		print('Simulating the marginalisation data set.')
		params  = np.array([[self.sample_prior(kk) for kk in self.param_names] for i in range(self.n_m)]).squeeze()
		self.sim_out_den = np.array([self.simulator(i) for i in params])

	def sim_numerator(self, theta, n_theta):
		self.sim_out_num = np.array([self.simulator(theta) for i in range(self.n_theta)])

	def ratio(self, theta, sim_out_num=None, get_score=False):
		n_m = self.sim_out_den.shape[0]
		if sim_out_num is None:
			self.sim_numerator(theta, self.n_theta)
		else:
			self.sim_out_num = sim_out_num
		sim_out_num, sim_out_den = self.sim_out_num, self.sim_out_den
		X = np.vstack((sim_out_num,sim_out_den))
		y = np.hstack((np.ones(sim_out_num.shape[0]),np.zeros(sim_out_den.shape[0])))

		clfy = LogisticRegressionCV(penalty=self.penalty, solver='saga', n_jobs=self.n_jobs) if self.clfy is None else self.clfy
		clfy.fit(X, y)

		sim_out_true = np.array([self.y_obs])
		#y_pred = clfy.predict(sim_out_true), 
		y_pred_prob  = clfy.predict_proba(sim_out_true).squeeze()
		n_theta, n_m = sim_out_num.shape[0], sim_out_den.shape[0]
		rr = (n_m/n_theta)*y_pred_prob[1]/y_pred_prob[0]
		rr = 1 if rr>1 else rr
		if get_score: return rr, clfy.score(X,y)
		return rr

	def run(self, thetas=None, n_grid_out=100, sim_out_num=None):
		if thetas is not None: self.thetas = thetas
		self.posterior = np.zeros(self.thetas.shape[0])
		for i, theta in enumerate(self.thetas):
			r0 = self.ratio(theta, sim_out_num=sim_out_num)
			self.posterior[i] = r0
			if self.verbose:
				if np.array(theta).size==1: theta = [theta]
				msg = ','.join(['{0:.3f}'.format(th) for th in theta]) 
				print('Pr({0:}) = {1:.5f}'.format(msg,r0))
				print('Completed: {0:.2f} %'.format(100*(i+1)/self.thetas.shape[0]))


class LFIRE_TrainingSetAuto:
	def __init__(self, simulator, observation, prior, bounds, n_m_init=10, n_theta_init=10, n_grid_out=25, thetas=None, verbose=True, penalty='l1', n_jobs=4, clfy=None):
		self.n_m_init = n_m_init
		self.n_theta_init = n_theta_init
		self.n_m = None
		self.n_theta = None
		lfi = LFIRE_core(simulator, observation, prior, bounds, sim_out_den=None, n_m=self.n_m_init, n_theta=self.n_theta_init, n_grid_out=100, thetas=thetas, verbose=verbose, penalty=penalty, n_jobs=n_jobs, clfy=clfy)

	def run(self, thetas=None, n_grid_out=100, sim_out_num=None):
		if thetas is not None: lfi.thetas = thetas
		lfi.posterior = np.zeros(lfi.thetas.shape[0])
		self.clfy_score = np.zeros(lfi.thetas.shape[0])
		for i, theta in enumerate(lfi.thetas):
			r0, s0 = lfi.ratio(theta, sim_out_num=sim_out_num, get_score=True)
			lfi.posterior[i] = r0
			self.clfy_score[i] = s0
			if self.verbose:
				if np.array(theta).size==1: theta = [theta]
				msg = ','.join(['{0:.3f}'.format(th) for th in theta]) 
				print('Pr({0:}) = {1:.5f}'.format(msg,r0))
				print('Completed: {0:.2f} %'.format(100*(i+1)/self.thetas.shape[0]))



class LFIRE_BayesianOpt:
	def __init__(self, simulator, observation, prior, bounds, sim_out_den=None, n_m=100, n_theta=100, n_grid_out=100, thetas=None, verbose=True, penalty='l1', n_jobs=4, clfy=None):
		#self.N_init  = N_init
		self.simulator = simulator
		#self.distance  = distance
		self.verbose = verbose
		self.penalty = penalty
		self.y_obs = observation
		self.param_names = [kk for kk in prior]
		self.param_bound = bounds
		self.bounds = np.array([bounds[kk] for kk in self.param_names])
		self.n_m = n_m
		self.n_theta = n_theta
		self.n_grid_out = n_grid_out
		self.n_jobs = n_jobs
		self.clfy    = clfy

		if sim_out_den is not None: 
			self.sim_out_den = sim_out_den
			self.n_theta = sim_out_den.shape[0]
		else: 
			self.sim_denominator()  

		if thetas is None: self.theta_grid()




LFIRE = LFIRE_core


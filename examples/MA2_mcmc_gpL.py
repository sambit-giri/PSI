import numpy as np 

def MA2(t1, t2, n_obs=100, batch_size=1, random_state=None):
    # Make inputs 2d arrays for numpy broadcasting with w
    t1 = np.asanyarray(t1).reshape((-1, 1))
    t2 = np.asanyarray(t2).reshape((-1, 1))
    random_state = random_state or np.random

    w = random_state.randn(batch_size, n_obs+2)  # i.i.d. sequence ~ N(0,1)
    x = w[:, 2:] + t1*w[:, 1:-1] + t2*w[:, :-2]
    return x

def autocov(x, lag=1):
    C = np.mean(x[:,lag:] * x[:,:-lag], axis=1)
    return C

# true parameters
t1_true = 0.6
t2_true = 0.2

y_obs = MA2(t1_true, t2_true)

# Plot the observed sequence
plt.figure(figsize=(11, 6));
plt.plot(y_obs.ravel(), c='k', ls='-', label='obs');

# To illustrate the stochasticity, let's plot a couple of more observations with the same true parameters:
plt.plot(MA2(t1_true, t2_true).ravel(), ls='-', label='obs1');
plt.plot(MA2(t1_true, t2_true).ravel(), ls='-', label='obs2');

plt.legend()



## ABC

def simulator_1(theta):
	t1, t2 = theta 
	x = MA2(t1, t2)
	return autocov(x)

def simulator(theta):
	out = np.array([])
	for i in range(theta.shape[0]):
		out = np.append(out, simulator_1(theta[i]))
	return out[:,None]

def distance(S1, S2):
	return ((S1-S2)**2).sum(axis=1)


abc = ABC_gpL(simulator, distance, y_obs, 
				theta_sampler=None, theta_range=theta_range, n_train_init=100,
				mcmc_sampler=None, mcmc_sampler_info=None
				)
abc.learn_logL()




def log_prior(theta):
	for i,ke in enumerate(abc.theta_range.keys()):
		if theta[i]<abc.theta_range[ke][0] or theta[i]>abc.theta_range[ke][1]:
			return -np.inf
	return 0.0

def log_probability(theta):
	if theta.ndim==1: theta = theta[None,:]
	lp = log_prior(theta)
	print(lp, type(theta), theta.shape)
	if not np.isfinite(lp):
		return -np.inf
	return lp + abc.logL_model.predict(theta)

pos_fn = lambda n=1: np.array([np.random.random(n)*(abc.theta_range[ke][1]-abc.theta_range[ke][0])+abc.theta_range[ke][0] for ke in abc.theta_range.keys()]).T
pos = pos_fn(abc.mcmc_sampler_info['nwalkers'])
nwalkers, ndim = pos.shape
n_samples = abc.mcmc_sampler_info['n_samples'] if 'n_samples' in abc.mcmc_sampler_info.keys() else 5**ndim

print(abc.mcmc_sampler_info['n_samples'])
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(), 
						pool=self.mcmc_sampler_info['pool'], 
						backend=self.mcmc_sampler_info['backend']
						)
sampler.run_mcmc(pos, self.mcmc_sampler_info['n_samples'], progress=True)
self.sampler = sampler

tau = sampler.get_autocorr_time()
print('Autocorrelation time:', tau)

flat_samples = sampler.get_chain(discard=tau*2, thin=1, flat=True)












theta_range = {'t1': [0,2], 't2': [0,2]} 
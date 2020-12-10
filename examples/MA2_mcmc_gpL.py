import numpy as np 
import matplotlib.pyplot as plt 

# import corner
# from psi import ABC_gpL

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
plt.show()



## ABC

#### 1 param

def simulator_1p_1(theta):
	t1, t2 = theta, t2_true
	x = MA2(t1, t2)
	return autocov(x)

def simulator_1p(theta):
	out = np.array([])
	for i in range(theta.shape[0]):
		out = np.append(out, simulator_1p_1(theta[i]))
	return out[:,None]

def distance_1p(S1, S2):
	if S1.ndim==1: S1 = S1[:,None]
	if S2.ndim==1: S2 = S2[:,None]
	return np.sqrt(((S1-S2)**2).sum(axis=1))


theta_true_1p  = np.array([t1_true])
y_obs_1p       = simulator_1p(np.array([t1_true])) 
theta_range_1p = {'t1': [-1,2]} 

print('Inferring 1 parameter.')

abc_1p = ABC_gpL(simulator_1p, distance_1p, y_obs_1p, 
				theta_sampler=None, theta_range=theta_range_1p, n_train_init=100,
				mcmc_sampler=None, mcmc_sampler_info=None
				)
abc_1p.create_dataset(5000)
abc_1p.learn_distance()
flat_samples_1p = abc_1p.run_mcmc(5000)  


labels_1p = ['t1']
fig_1p = corner.corner(
    flat_samples_1p, labels=labels_1p, truths=theta_true_1p
);

plt.show()

### 2 param

def simulator_1(theta):
	t1, t2 = theta 
	x = MA2(t1, t2)
	return autocov(x)

def simulator(theta):
	if theta.ndim==1:
		return simulator_1(theta)
	out = np.array([])
	for i in range(theta.shape[0]):
		out = np.append(out, simulator_1(theta[i]))
	return out[:,None]

def distance(S1, S2):
	return ((S1-S2)**2).sum(axis=1)


theta_true = np.array([t1_true, t2_true])
y_obs = simulator(theta_true) 
theta_range = {'t1': [-1,2], 't2': [-1,2]} 

print('Inferring 2 parameter.')

abc = ABC_gpL(simulator, distance, y_obs, 
				theta_sampler=None, theta_range=theta_range, n_train_init=100,
				mcmc_sampler=None, mcmc_sampler_info=None
				)
abc.learn_distance(1000)
flat_samples = abc.run_mcmc(50000)  


labels = ['t1', 't2']
fig = corner.corner(
    flat_samples, labels=labels, truths=theta_true
)

plt.show()










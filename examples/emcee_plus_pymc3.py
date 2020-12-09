import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

true_params = np.array([0.5, -2.3, -0.23])

N = 50
t = np.linspace(0, 10, 2)
x = np.random.uniform(0, 10, 50)
y = x * true_params[0] + true_params[1]
y_obs = y + np.exp(true_params[-1]) * np.random.randn(N)

plt.plot(x, y_obs, ".k", label="observations")
plt.plot(t, true_params[0]*t + true_params[1], label="truth")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(fontsize=14);


import pymc3 as pm
import theano.tensor as tt

with pm.Model() as model:
    logs = pm.Uniform("logs", lower=-10, upper=10)
    alphaperp = pm.Uniform("alphaperp", lower=-10, upper=10)
    theta = pm.Uniform("theta", -2*np.pi, 2*np.pi, testval=0.0)

    # alpha_perp = alpha * cos(theta)
    alpha = pm.Deterministic("alpha", alphaperp / tt.cos(theta))
    
    # beta = tan(theta)
    beta = pm.Deterministic("beta", tt.tan(theta))
    
    # The observation model
    mu = alpha * x + beta
    pm.Normal("obs", mu=mu, sd=tt.exp(logs), observed=y_obs)
    
    trace = pm.sample(draws=2000, tune=2000)



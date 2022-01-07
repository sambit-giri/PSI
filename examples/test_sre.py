import numpy as np
from importlib import reload
import psi
import emcee 
from chainconsumer import ChainConsumer
from tqdm import tqdm 

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline

yerr_param = [-1, 2.0]#[0.1, 1.5]
line  = psi.sample_models.noisy_line(yerr_param=yerr_param)
xs    = line.xs()
y_obs = line.observation()

##################### 2 param, slope, intercept ######################

simulator = lambda x: line.simulator(x[0], x[1])[None,:] if x.ndim==1 else np.array([line.simulator(i[0], i[1]) for i in x])

prior  = {'m': 'uniform', 'c': 'uniform'}
bounds = {'m': [-2.5, 0.5], 'c': [0,10]}

def log_prior(theta):
	if bounds['m'][0]<=theta[0]<=bounds['m'][1] and bounds['c'][0]<=theta[1]<=bounds['c'][1]: 
		return 0
	return -np.inf

def logL(theta):
	lp = log_prior(theta)
	if not np.isfinite(lp): return lp 
	y_mod = theta[0]*xs + theta[1]
	return -0.5*np.sum((y_mod-y_obs)**2)

mins = np.array([-2.5,0])
maxs = np.array([0.5,10])

pos = mins+(maxs-mins) * np.random.uniform(0,1,size=(64, len(mins)))
nwalkers, ndim = pos.shape
n_samples = 1000
sampler = emcee.EnsembleSampler(nwalkers, ndim, logL)
sampler.run_mcmc(pos, n_samples, progress=True);
flat_samples = sampler.get_chain(discard=0, flat=True) 

c = ChainConsumer()
labels = [ke for ke in bounds.keys()]
theta_true = [line.true_slope, line.true_intercept]
c.add_chain(flat_samples, parameters=labels)
fig = c.plotter.plot(figsize="column", truth=theta_true)
fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
plt.show()







from sklearn.pipeline import Pipeline

clf = Pipeline([('normalize', StandardScaler()), ('classifier', MLPClassifier(alpha=1, max_iter=1000))])
lfi = emceeSRE()
lfi.set_simulator(simulator)
lfi.learn_logL_with_classifier(clf, mins, maxs, Nsamples=10000, test_size=0.1)
lfi.set_obs(y_obs)

from skorch import NeuralNetClassifier
import torch
from torch import nn
from torch.nn import functional as F

class ClassifierModule(nn.Module):
    def __init__(
            self,
            num_units=50,
            nonlin=F.relu,
            dropout=0.5,
    ):
        super(ClassifierModule, self).__init__()
        self.num_units = num_units
        self.nonlin = nonlin
        self.dropout = dropout

        self.dense0 = nn.Linear(y_obs.shape[1]+2, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(dropout)
        self.dense1 = nn.Linear(num_units, 20)
        self.output = nn.Linear(20, 2)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.softmax(self.output(X), dim=-1)
        return X


model = NeuralNetClassifier(
    ClassifierModule,
    max_epochs=1000,
    lr=0.1,
    # optimizer=torch.optim.SGD,
    # criterion=nn.BCELoss,
    iterator_train__shuffle=True,
)


lfi = emceeSRE()
lfi.set_simulator(simulator)
lfi.learn_logL_with_classifier(model, mins, maxs, Nsamples=10000, test_size=0.1)
lfi.set_obs(y_obs)


from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=50)
# clf = Pipeline([('normalize', StandardScaler()), ('classifier', KNeighborsClassifier(n_neighbors=30) )]) 
lfi = emceeSRE()
lfi.set_simulator(simulator)
lfi.learn_logL_with_classifier(clf, mins, maxs, Nsamples=10000, test_size=0.1)
lfi.set_obs(y_obs)


def logL(theta):
	lp = log_prior(theta)
	if not np.isfinite(lp): return lp 
	return lp+lfi.logL_estimator(theta)

pos = mins+(maxs-mins) * np.random.uniform(0,1,size=(64, len(mins)))
nwalkers, ndim = pos.shape
n_samples = 1000
sampler = emcee.EnsembleSampler(nwalkers, ndim, logL)
sampler.run_mcmc(pos, n_samples, progress=True);
flat_samples = sampler.get_chain(discard=0, flat=True) 


from sklearn import svm

clf = svm.SVC(kernel='linear', probability=True)
lfi = emceeSRE()
lfi.set_simulator(simulator)
lfi.learn_logL_with_classifier(clf, mins, maxs, Nsamples=10000, test_size=0.1)
lfi.set_obs(y_obs)


from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(n_estimators=100, random_state=0)
lfi = emceeSRE()
lfi.set_simulator(simulator)
lfi.learn_logL_with_classifier(clf, mins, maxs, Nsamples=10000, test_size=0.1)
lfi.set_obs(y_obs)


c = ChainConsumer()
labels = [ke for ke in bounds.keys()]
theta_true = [line.true_slope, line.true_intercept]
c.add_chain(flat_samples, parameters=labels)
fig = c.plotter.plot(figsize="column", truth=theta_true)
fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
plt.show()


ni = 50
mesh = np.array([[i,j] for i in np.linspace(mins[0],maxs[0],ni) for j in np.linspace(mins[1],maxs[1],ni)])
logLmesh = np.array([logL(m) for m in tqdm(mesh)])
plt.pcolor(mesh[:,0].reshape(ni,ni),mesh[:,1].reshape(ni,ni),logLmesh.reshape(ni,ni))





############## https://elfi.readthedocs.io/en/latest/quickstart.html

import elfi 
import scipy.stats as ss


mu = elfi.Prior('uniform', -2, 4)
sigma = elfi.Prior('uniform', 1, 4)

def simulator(theta, batch_size=1, random_state=None):
	if type(theta)==list: theta = np.array(theta)
	if theta.ndim>1:
		out = [simulator(tt, batch_size=batch_size, random_state=random_state) for tt in theta]
		return np.array(out).squeeze()
	mu, sigma = theta
	mu, sigma = np.atleast_1d(mu, sigma)
	return ss.norm.rvs(mu[:, None], sigma[:, None], size=(batch_size, 30), random_state=random_state)

def mean(y):
    return np.mean(y, axis=1)

def var(y):
    return np.var(y, axis=1)

# Set the generating parameters that we will try to infer
mean0 = 1
std0 = 3

# Generate some data (using a fixed seed here)
np.random.seed(20170525)
y_obs = simulator([mean0, std0])
print(y_obs)

def log_prior(theta):
	if -2<=theta[0]<=4 and 1<=theta[1]<=4: 
		return 0
	return -np.inf

def logL(theta):
	lp = log_prior(theta)
	if not np.isfinite(lp): return lp 
	return lp+lfi.logL_estimator(theta)


mins = np.array([-2, 1])
maxs = np.array([ 4, 4])

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, random_state=0, verbose=1, n_jobs=4)
lfi = emceeSRE()
lfi.set_simulator(simulator)
lfi.learn_logL_with_classifier(clf, mins, maxs, Nsamples=50000, test_size=0.1)
lfi.set_obs(y_obs)

ni = 50
mesh = np.array([[i,j] for i in np.linspace(mins[0],maxs[0],ni) for j in np.linspace(mins[1],maxs[1],ni)])
logLmesh = np.array([logL(m) for m in tqdm(mesh)])
plt.pcolor(mesh[:,0].reshape(ni,ni),mesh[:,1].reshape(ni,ni),logLmesh.reshape(ni,ni))



from skorch import NeuralNetClassifier
import torch
from torch import nn
from torch.nn import functional as F

class ClassifierModule(nn.Module):
    def __init__(
            self,
            num_units=50,
            nonlin=F.relu,
            dropout=0.5,
    ):
        super(ClassifierModule, self).__init__()
        self.num_units = num_units
        self.nonlin = nonlin
        self.dropout = dropout

        self.dense0 = nn.Linear(y_obs.shape[1]+2, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(dropout)
        self.dense1 = nn.Linear(num_units, 20)
        self.output = nn.Linear(20, 2)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.sigmoid(self.output(X)) #F.softmax(self.output(X), dim=-1)
        return X


model = NeuralNetClassifier(
    ClassifierModule,
    max_epochs=1000,
    lr=0.01,
    # optimizer=torch.optim.SGD,
    criterion=nn.CrossEntropyLoss, #criterion=nn.NLLLoss, # criterion=nn.BCELoss,
    # iterator_train__shuffle=True,
)

model.fit(X,y)


lfi = emceeSRE()
lfi.set_simulator(simulator)
lfi.learn_logL_with_classifier(model, mins, maxs, Nsamples=10000, test_size=0.1)
lfi.set_obs(y_obs)





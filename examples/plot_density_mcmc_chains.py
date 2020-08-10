import numpy as np
import matplotlib.pyplot as plt
from importlib import reload 

import psi
from psi import corner_mcmc_chains
%pylab

# Create some random samples for demonstration:
# make random covariance, then independent samples from Gaussian

ndim = 4
nsamp = 10000
np.random.seed(10)
A = np.random.rand(ndim,ndim)
cov = np.dot(A, A.T)
samps1 = np.random.multivariate_normal([0]*ndim, cov, size=nsamp)
A = np.random.rand(ndim,ndim)
cov = np.dot(A, A.T)
samps2 = np.random.multivariate_normal([0]*ndim, cov, size=nsamp)

# Triangle plot of one distribution
corner_mcmc_chains.plot_corner_dist(samps1)


def plot_corner_multiple_dist(samples, labels=None, flavor='hist', bins_1d=60, bins_2d=60, shading='gouraud', linestyle='-', linewidth=2, normed=True, parallise=False, CI=[68,95], CI_plotparam=None, smooth_dist=2.5):
	assert isinstance(samples, (list,tuple))

	cmap_options = ['Blues', 'Oranges', 'Greys', 'Purples', 'Greens', 'Reds']
	colr_options = ['blue', 'orange', 'grey', 'purple', 'green', 'red']

	n_samples = samples.shape[1]
	if labels is None: labels = ['$\\theta_%d$'%i for i in range(n_samples)]
	else: assert len(labels)==n_samples
	fig, axes = plt.subplots(ncols=n_samples, nrows=n_samples, figsize=(10,8))
	fig.subplots_adjust(left=0.15, bottom=0.12, right=0.90, top=0.96, wspace=0.1, hspace=0.1)
	if np.array(bins_1d).size==1: bins_1d = [bins_1d for i in range(n_samples)]
	if np.array(bins_2d).size==1: bins_2d = [bins_2d for i in range(n_samples)]

	for i in range(n_samples):
		for j in range(n_samples):
			print(i+1,j+1)
			if j>i: axes[i,j].set_visible(False)
			else:
				if i==j: 
					density_1D(samples[:,i], axes=axes[i,j], bins=bins_1d[i], linestyle=linestyle, linewidth=linewidth, normed=normed, smooth_dist=smooth_dist)
				else: 
					im = density_2D(samples[:,j], samples[:,i], CI=CI, axes=axes[i,j], flavor=flavor, nbins=bins_2d[i], cmap=cmap, shading=shading, CI_plotparam=CI_plotparam, smooth_dist=smooth_dist)
			if j==0: 
				if i!=0: 
					axes[i,j].set_ylabel(labels[i])
				else: 
					axes[i,j].set_yticks([])
			else: 
				axes[i,j].set_yticks([])
			if i==n_samples-1: 
				axes[i,j].set_xlabel(labels[j])
			else: 
				axes[i,j].set_xticks([])
	#fig.set_size_inches(18.5, 10.5)
	fig.subplots_adjust(right=0.9)
	cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
	fig.colorbar(im, cax=cbar_ax)
	### Estimate the CI
	print_CI_samples(samples, bins_1d=bins_1d, CI=CI, labels=labels, smooth_dist=smooth_dist)
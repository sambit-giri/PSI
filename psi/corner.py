import numpy as np 
import matplotlib.pyplot as plt
from skimage.filters import gaussian

def plot_lfire(lfi, smooth=5):
	if np.ndim(lfi.thetas)==1:
		fig, axes = plt.subplots(nrows=1, ncols=1) 
		axes.plot(lfi.thetas, lfi.posterior)
		axes.set_xlabel(lfi.param_names[0])
		plt.show()
	else:
		N = lfi.thetas.shape[1]
		fig, axes = plt.subplots(nrows=N, ncols=N)
		for i in range(N):
			for j in range(N):
				if j>i: axes[i,j].axis('off')
				elif i==j: 
					plot_1Dmarginal(lfi, i, ax=axes[i,j], smooth=smooth)
					if i+1<N: 
						axes[i,j].set_xlabel('')
						axes[i,j].set_xticks([])
					if j>0:
						axes[i,j].set_yticks([])
				else: 
					im = plot_2Dmarginal(lfi, i, j, ax=axes[i,j], smooth=smooth)
					if i+1<N: 
						axes[i,j].set_xlabel('')
						axes[i,j].set_xticks([])
					if j>0: 
						axes[i,j].set_ylabel('')
						axes[i,j].set_yticks([])

	fig.subplots_adjust(right=0.88)
	cb_ax = fig.add_axes([0.89, 0.1, 0.02, 0.8])
	cbar = fig.colorbar(im, cax=cb_ax)
	plt.show()


def plot_2Dmarginal(lfi, idx, idy, ax=None, bins=100, verbose=False, smooth=False):
	N = lfi.thetas.shape[1]
	thetas = lfi.thetas
	inds = np.arange(N); inds = np.delete(inds, idx); inds = np.delete(inds, idy)
	X = np.array([thetas[:,i] for i in inds])
	X = np.vstack((thetas[:,idx].reshape(1,-1), thetas[:,idy].reshape(1,-1))).T if X.size==0 else np.vstack((thetas[:,idx].reshape(1,-1), thetas[:,idy].reshape(1,-1), X)).T
	y = lfi.posterior
	dm   = [int(np.round(y.shape[0]**(1/X.shape[1]))) for i in range(X.shape[1])] 
	cube = y.reshape(dm)
	cube = np.swapaxes(cube, 0, idx)
	cube = np.swapaxes(cube, 1, idy)
	while(cube.ndim>2):
		if verbose: print('Reducing dimension: {0:} -> {1:}'.format(cube.ndim,cube.ndim-1))
		cube = cube.sum(axis=-1)
	yy = np.unique(X[:,0])
	xx = np.unique(X[:,1])
	xi, yi = np.meshgrid(xx,yy)
	if smooth: cube = gaussian(cube, smooth) 
	if ax is None: fig, ax = plt.subplots(nrows=1, ncols=1)
	im = ax.pcolormesh(xi, yi, (cube-cube.min())/(cube.max()-cube.min()), cmap='Blues')
	if ax is None: fig.colorbar(im, ax=ax)
	#ax.imshow(xx, cube)
	ax.set_xlabel(lfi.param_names[idy])
	ax.set_ylabel(lfi.param_names[idx])
	return im


def plot_1Dmarginal(lfi, idx, ax=None, bins=100, verbose=False, smooth=False):
	N = lfi.thetas.shape[1]
	thetas = lfi.thetas
	inds = np.arange(N); inds = np.delete(inds, idx)
	X    = np.array([thetas[:,i] for i in inds])
	X    = np.vstack((thetas[:,idx].reshape(1,-1), X)).T
	y    = lfi.posterior
	dm   = [int(np.round(y.shape[0]**(1/X.shape[1]))) for i in range(X.shape[1])] 
	cube = y.reshape(dm)
	if idx!=0: cube = np.swapaxes(cube, 0, idx)
	while(cube.ndim>1):
		if verbose: print('Reducing dimension: {0:} -> {1:}'.format(cube.ndim,cube.ndim-1))
		cube = cube.sum(axis=-1)
	xx = np.unique(X[:,0])
	if smooth: cube = gaussian(cube, smooth) 
	if ax is None:
		fig, ax = plt.subplots(nrows=1, ncols=1)
	ax.plot(xx, (cube-cube.min())/(cube.max()-cube.min()))
	ax.set_xlabel(lfi.param_names[idx])


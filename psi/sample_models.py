import numpy as np

class gaussian_signal:
	def __init__(self, true_mean, true_sigma):
		self.true_mean  = true_mean
		self.true_sigma = true_sigma

	def observation(self, N=20):
		return np.random.normal(loc=self.true_mean, scale=self.true_sigma, size=N)

	def simulator(self, mean, sigma, N=1):
		return np.random.normal(loc=mean, scale=sigma, size=N)

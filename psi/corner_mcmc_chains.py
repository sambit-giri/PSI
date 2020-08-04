import matplotlib.pyplot as plt
from getdist import plots, MCSamples
import getdist

def plot_dist_corner(samples, labels=None, names=None, filled=True, sample_labels=None):
	if isinstance(samples, (list,tuple)):
		if sample_labels is None: sample_labels = ['samples {0:d}'.format(i) for i in range(len(samples))]
		samps = [MCSamples(samples=samples0, names=names, labels=labels, label=str(slabels0)) for samples0, slabels0 in zip(samples,sample_labels)]
		g = plots.get_subplot_plotter()
		g.triangle_plot(samps, filled=filled)
	else:
		samps = MCSamples(samples=samples, names=names, labels=labels)
		g = plots.get_subplot_plotter()
		g.triangle_plot(samps, filled=filled)

	return None

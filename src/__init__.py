'''
Tools21cm is a Python package for analysing simulations of the 21-cm signal
during Epoch of Reionization and Cosmic Dawn.
We incorpoarte its predecessor, c2raytools, into this package.

You can also get documentation for all routines directory from
the interpreter using Python's built-in help() function.
For example:
>>> import tools21cm as t2c
>>> help(t2c.calc_dt)
'''

import sys

from . import sample_models
from . import distances
from . import bolfi
from . import bayesian_optimisation
from . import lfire
from . import rejectionABC
from . import corner
# from . import corner_mcmc_chains
from . import kernel_density

from . import infer_sampler

#Suppress warnings from zero-divisions and nans
import numpy
numpy.seterr(all='ignore')

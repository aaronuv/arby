import arby
import numpy as np
from scipy.special import jv as BesselJ


npoints = 1001
nsamples = 10001

# Sample parameter nu and variable x
nu = np.linspace(1, 5, num=npoints)
x = np.linspace(0, 100, num=nsamples)

# build traning set
training = np.array([BesselJ(nn, x) for nn in nu])

# function call
import statprof
with statprof.profile():
    rb_data = arby.reduced_basis(training_set=training, 
                physical_points=x, greedy_tol=1e-12)
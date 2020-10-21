# --- greedy.py ---

"""
	Classes for building reduced basis greedy algorithms
"""

__author__ = "Chad Galley <crgalley@tapir.caltech.edu, crgalley@gmail.com>"

import numpy as np
import lib


#############################################
# Class for iterated, modified Gram-Schmidt #
#      orthonormalization of functions      #
#############################################

class _IteratedModifiedGramSchmidt(object):
  """Iterated modified Gram-Schmidt algorithm for building an orthonormal basis.
  Algorithm from Hoffman, `Iterative Algorithms for Gram-Schmidt Orthogonalization`.
  """
  
  def __init__(self, inner):
    self.inner = inner
  
  def add_basis(self, h, basis, a=0.5, max_iter=3):
    """Given a function, h, find the corresponding basis function orthonormal to all previous ones"""
    norm = self.inner.norm(h)
    e = h/norm
    #r = np.zeros(len(basis))
    flag, ctr = 0, 1
    while flag == 0:
      #s = np.zeros(len(basis))
      for b in basis:
      #for ii, bb in enumerate(basis):
        # s[ii] = self.inner.dot(bb, e, self.inner_type)
        # e -= bb*s[ii]
        e -= b*self.inner.dot(b, e)
      #r += s
      new_norm = self.inner.norm(e)
      #r[0] = new_norm
      # Iterate, if necessary
      if new_norm/norm <= a:
        norm = new_norm
        ctr += 1
        if ctr > max_iter:
          print(">>> Warning(Max number of iterations reached) Basis may not be orthonormal.")
          flag = 1
      else:
        flag = 1
    
    #print r
    
    return [e/new_norm, new_norm]
    #return [e/new_norm, new_norm, r]
    
  def make_basis(self, hs, norms=False, a=0.5, max_iter=3):
    """Given a set of functions, hs, find the corresponding orthonormal set of basis functions."""
    
    dim = np.shape(hs)
    #basis = np.zeros(dim, dtype=hs.dtype)
    basis = np.zeros_like(hs)
    basis[0] = self.inner.normalize(hs[0])
    # R = np.zeros((dim[0],dim[0]), dtype=hs.dtype)
    # R[0][0] = self.inner.norm(hs[0], self.inner_type)
    if norms:
      norm = np.zeros(dim[0], dtype='double')
      norm[0] = self.inner.norm(hs[0])
    
    for ii in range(1, dim[0]):
      if norms:
        basis[ii], norm[ii] = self.add_basis(hs[ii], basis[:ii], a=a, max_iter=max_iter)
        #basis[ii], norm[ii], R[ii-1][:ii] = self.add_basis(hs[ii], basis[:ii], a=a, max_iter=max_iter)
      else:
        basis[ii], _ = self.add_basis(hs[ii], basis[:ii], a=a, max_iter=max_iter)
        #basis[ii], _, R[ii-1][:ii] = self.add_basis(hs[ii], basis[:ii], a=a, max_iter=max_iter)
            
    if norms:
      return [np.array(basis), norm]
      #return [np.array(basis), norm, R]
    else:
      return np.array(basis)
      #return [np.array(basis), R]


class GramSchmidt(_IteratedModifiedGramSchmidt):
  """Class for building an orthonormal basis using the
  iterated, modified Gram-Schmidt procedure.
  
  Input
  -----
  vectors    -- set of vectors to orthonormalize
  inner      -- instance of Integration class
  normsQ     -- norms of input vectors (default is False)
  
  Methods
  -------
  iter -- one iteration of the iterated, modified
          Gram-Schmidt algorithm
  make -- orthonormalize all the input vectors
  
  Examples
  --------
  Create an instance of the Basis class for functions with
  unit norm::
  
  >>> basis = rp.algorithms.Basis(vectors, inner)
  
  Build an orthonormal basis by running
  
  >>> basis.make()
  
  Output is an array of orthonormal basis elements.  
  """
  
  def __init__(self, vectors, integration, normsQ=False):
    self.Nbasis, self.Nnodes = np.shape(vectors)
    self.functions = np.asarray(vectors)
    
    _IteratedModifiedGramSchmidt.__init__(self, integration)
    
    self.normsQ = normsQ
    if self.normsQ:
      self.norms = lib.malloc(self.functions.dtype, self.Nbasis)
    
    self.basis = lib.malloc(self.functions.dtype, self.Nbasis, self.Nnodes)
  
  def iter(self, step, h, a=0.5, max_iter=3):
    """One iteration of the iterated, modified Gram-Schmidt algorithm"""
    ans = self.add_basis(h, self.basis[:step], a=a, max_iter=max_iter)
    
    if self.normsQ:
      #self.basis[step+1], self.norms[step+1] = ans
      self.basis[step], self.norms[step+1] = ans
    else:
      #self.basis[step+1], _ = ans
      self.basis[step], _ = ans
  
  def make(self, a=0.5, max_iter=3, timerQ=False):
    """Find the corresponding orthonormal set of basis functions."""
    
    self.basis[0] = self.inner.normalize(self.functions[0])
    if self.normsQ:
      self.norms[0] = self.inner.norm(self.functions[0])
    
    if timerQ:
      t0 = time.time()
    
    for ii in range(1, self.Nbasis):
      self.iter(ii, self.functions[ii], a=a, max_iter=max_iter)
    
    if timerQ:
      print("\nElapsed time =", time.time()-t0)
    
    if self.normsQ:
      return [np.array(self.basis), self.norms]
    else:
      return np.array(self.basis)


#############################################
# K-fold cross-validation for reduced basis #
#############################################

def _Kfold(training, name, K, i_fold, folds, **options):
  """K-fold cross validation on a given partition (or fold)"""
  
  # Assemble training data excluding the i_fold'th partition for validation
  # TODO: There's got to be a better way to do this
  folds = np.asarray(folds)
  complement = np.ones(len(folds), dtype='bool')
  complement[i_fold] = False
  f_trial = np.sort( np.concatenate([ff for ff in folds[complement]]) )
  fold = folds[i_fold]

  # Form trial data
  # dims, size = get_dims(np.shape(x), np.shape(y))
#   if dims == 1:
#     x_trial = x[f_trial]
#     x_test = x[fold]
#   elif dims == 2:
#     x_trial = x[:,f_trial]
#     x_test = x[:, fold]
#   else:
#     raise Exception, "Expecting 1d or 2d arrays."
#   y_trial = y[f_trial]
#   y_test = y[fold]
#
#   # Build trial interpolant
#   if fit == 'romSpline':
#     if _romSpline:
#       interp = romSpline._greedy(x_trial, y_trial, **options)[0]
#     else:
#       raise Exception, "romSpline module not imported."
#   else:
#     interp = _dict[fit](x_trial, y_trial, **options)[0]
  
  rb = greedy.ReducedBasis()
  
  # Compute L-infinity errors between trial and data in validation partition
  error, arg_error = _Linfty(interp(x_test), y_test, arg=True)
  
  return error, fold[arg_error]


class CrossValidation(object):
  
  def __init__(self, **options):
    self._options = options
    
    self.errors = {}
    self.arg_errors = {}
    self.ensemble_errors = {}
    self.ensemble_arg_errors = {}
  
  
  def __call__(self, x, y, K=10, parallel=True, random=True):
    return self.Kfold(x, y, K=K, parallel=parallel, random=random)
  
  
  def LOOCV(self, x, y, parallel=True, random=True):
    """Leave-one-out cross-validation"""
    self.Kfold(x, y, K=np.shape(x)[-1], parallel=parallel, random=random)
  
  
  #def Kfold(self, x, y, K=10, parallel=True, random=True):
  def Kfold(self, training, name, K=10, parallel=True, random=True):
    """K-fold cross-validation"""
    
    # Validate input
    assert type(name) is str, "Expecting string input for `name`"
    
    # Allocate some memory
    errors = np.zeros(K, dtype='double')
    arg_errors = np.zeros(K, dtype='int')
    
    # Divide into K random or nearly equal partitions
    if random:
      self._partitions = random_partitions(size, K)
    else:
      self._partitions = partitions(size, K)
    
    if parallel is False:
      for ii in range(K):
        errors[ii], arg_errors[ii] = _Kfold(training, name, K, ii, self._partitions, **self._options)
    else:
      # Determine the number of processes to run on
      if parallel is True:
        try:
          numprocs = cpu_count()
        except NotImplementedError:
          numprocs = 1
      else:
        numprocs = parallel
      
      # Create a parallel process executor
      executor = ProcessPoolExecutor(max_workers=numprocs)
      
      # Compute the interpolation errors on each testing partition
      output = []
      for ii in range(K):
        output.append( executor.submit(_Kfold, training, name, K, ii, self._partitions, **self._options) )
      
      # Gather the results as they complete
      for ii, ee in enumerate(as_completed(output)):
        errors[ii], arg_errors[ii] = ee.result()
      
    self.errors[K] = errors
    self.arg_errors[K] = arg_errors
  
  
  def Kfold_ensemble(self, x, y, n, K=10, parallel=True, random=True, verbose=False):
    """Perform a number n of K-fold cross-validations"""
    ens_arg_errors, ens_errors = [], []
    for nn in range(n):
      if verbose and not (nn+1)%10:
        print("Trial number", nn+1)
      self.Kfold(x, y, K=K, parallel=parallel, random=random)
      ens_errors.append( self.errors[K] )
      ens_arg_errors.append( self.arg_errors[K] )
    
    self.ensemble_errors[K] = ens_errors
    self.ensemble_arg_errors[K] = ens_arg_errors


#############################################
# Class for reduced basis greedy algorithms #
#############################################

class _ReducedBasis(object):
  
  def __init__(self, inner):
    self.inner = inner
  
  def malloc(self, Nbasis, Npoints, Nquads, Nmodes=1, dtype='complex'):
    """Allocate memory for numpy arrays used for making reduced basis"""
    self.errors = lib.malloc('double', Nbasis)
    self.indices = lib.malloc('int', Nbasis)
    if Nmodes == 1:
      self.basis = lib.malloc(dtype, Nbasis, Nquads)
    elif Nmodes > 1:
      self.basis = lib.malloc(dtype, Nbasis, Nmodes, Nquads)
    else:
      raise Exception("Expected positive number of modes.")
    self.basisnorms = lib.malloc('double', Nbasis)
    self.alpha = lib.malloc(dtype, Nbasis, Npoints)	
    
  def _alpha(self, e, h):
    """Inner product of a basis function e with a function h:
        alpha(e,h) = <e, h>
    """
    return self.inner.dot(e, h)
    
  def alpha_arr(self, e, hs):
    """Inner products of a basis function e with an array of functions hs"""
    return np.array([self._alpha(e, hh) for hh in hs])
  
  def proj_error_from_basis(self, basis, h):
    """Square of the projection error of a function h on basis"""
    norm = self.inner.norm(h).real
    dim = len(basis[:,0])
    ans = 0.
    for ii in range(dim):
      ans += np.abs(self._alpha(basis[ii], h))**2
    return norm**2-ans
    #return norm**2-np.sum(np.abs(self._alpha(basis[ii], h))**2 for ii in range(dim))
    
  def proj_errors_from_basis(self, basis, hs):
    """Square of the projection error of functions hs on basis"""
    #norms = np.real([self.inner.norm(hh, self.inner_type) for hh in hs])
    #dim = len(basis[:,0])
    #return [norms**2-np.sum(np.abs(self._alpha(basis[ii], hh))**2 for ii in range(dim)) for hh in hs]
    return [self.proj_error_from_basis(basis, hh) for hh in hs]
    
  def proj_mismatch_from_basis(self, basis, h):
    """Mismatch of a function h with its projection onto the basis"""
    norms = self.inner.norm(h).real
    dim = len(basis[:,0])
    return 1.-(np.sum(abs(self._alpha(basis[ii], h))**2 for ii in range(dim)).real)/norms
  
  def proj_errors_from_alpha(self, alpha, norms=None):
    """Square of the projection error of a function h on basis in terms of pre-computed alpha matrix"""
    if norms is None:
      norms = np.ones(len(alpha[0]), dtype='double')
    #dim = len(alpha[:,0])
    ans = 0.
    for aa in alpha:
      ans += np.abs(aa)**2
    return norms**2 - ans
    #return np.sqrt(np.abs(norms**2 - ans))
    #return norms**2-np.sum(abs(alpha[ii])**2 for ii in range(dim))
  
  def projection_from_basis(self, h, basis):
    """Project a function h onto the basis functions"""
    #return np.array([ee*self._alpha(ee, h) for ee in basis])
    ans = 0.
    for ee in basis:
      ans += ee*self._alpha(ee, h)
    return ans
  
  def projection_from_alpha(self, alpha, basis):
    """Project a function h onto the basis functions using the precomputed
    quantity alpha = <basis, h>"""
    ans = 0.
    for ii, ee in basis:
      ans += ee*alpha[ii]
    return ans
  
  def _Alpha(self, E, e, alpha):
    return self.inner.dot(E, self.projection_from_alpha(alpha, e))
  
  def Alpha_arr(self, E, e, alpha):
    return np.array([self._Alpha(EE, e, alpha) for EE in E])
  
  def partition_proj_errors_from_alpha(self, E, e, alpha):
    A = self.Alpha_arr(E, e, alpha)
    return np.sum(np.abs(aa)**2 for aa in alpha) - np.sum(np.abs(AA)**2 for AA in A)
  
  


class ReducedBasis(_ReducedBasis, _IteratedModifiedGramSchmidt):
  """Class for standard reduced basis greedy algorithm.
  
  Input
  -----
  inner  -- method of InnerProduct instance
  
  Methods
  ---------
  seed -- seed the greedy algorithm
  iter -- one iteration of the greedy algorithm
  make -- implement the greedy algorithm from beginning to end
  trim -- trim zeros from remaining allocated entries
  
  Examples
  --------
  Create a ReducedBasis object for functions with unit norm::
  
  >>> rb = rp.ReducedBasis(inner)
  
  Let T be the training space of functions, 0 be the seed index, 
  and 1e-12 be the tolerance. The standard reduced basis greedy 
  algorithm is::
  
  >>> rb.seed(0, T)
  >>> for i in range(Nbasis):
  >>> ...if rb.errors[i] <= 1e-12:
  >>> ......break
  >>> ...rb.iter(i,T)
  >>> rb.trim(i)
  
  For convenience, this algorithm is equivalently implemented in 
  `make`::
  
  >>> rb.make(T, 0, 1e-12)
  
  Let T' be a different training space. The greedy algorithm can
  be run again on T' using::
  
  >>> rb.make(T', 0, 1e-12)
  
  or, alternatively, at each iteration using::
  
  >>> ...rb.iter(i,T')
  
  in the for-loop above.
  """
  
  def __init__(self, inner=None, loss='L2'):
    """
    loss -- the loss function to use for measuring the error
         between training data and its projection onto the
         reduced basis
         (default is 'L2' norm)
    """
    
    if inner is not None:
      self.inner = inner
      _ReducedBasis.__init__(self, inner)
      _IteratedModifiedGramSchmidt.__init__(self, inner)
      
      # Set the loss function that measures the discrepancy between
      # the training space data and its projection onto the reduced basis
      assert type(loss) is str, "Expecting string for variable `loss`."
      self._loss = loss
      if loss == 'L2':
        self.loss = self.proj_errors_from_alpha
      if loss == 'Linfty':
        def Linfty(alpha, basis, training):
          num = len(training)
          projs = np.dot(alpha.T, basis) # TODO: Don't use .T if possible
          #return np.array([self.inner.Linfty(training[nn]-projs[nn]) for nn in range(num)])
          #return self.inner.Linfty(training-projs)
          return np.array([self.inner.Linfty(training[ii]-projs[ii]) for ii in range(num)])
        self.loss = Linfty
    else:
      print("No integration rule given.")
  
  
  def seed(self, Nbasis, training_space, seed):
    """Seed the greedy algorithm.
    
    Seeds the first entries in the errors, indices, basis, and alpha arrays 
    for use with the standard greedy algorithm for producing a reduced basis 
    representation.
    
    Input
    -----
    Nbasis         -- number of requested basis vectors to make
    training_space -- the training space of functions
    seed           -- array index for seed point in training set
    
    Examples
    --------
    
    If rb is an instance of StandardRB, 0 is the array index associated
    with the seed, and T is the training set then do::
    
    >>> rb.seed(0, T)
      
    """
    
    # Extract dimensions of training space data
    dim = np.shape(np.asarray(training_space))
    if len(dim) == 2:
      Npoints, Nsamples = dim
      Nmodes = 1
    elif len(dim) == 3:
      Npoints, Nmodes, Nsamples = dim
    else:
      raise Exception("Unexpected dimensions for training space.")
    
    # Compute norms of training space data
    self._norms = np.array([self.inner.norm(tt) for tt in training_space])
    
    # Validate inputs
    assert Nsamples == np.size(self.inner.weights), "Number of samples is inconsistent with quadrature rule."
    self._Nbasis = Nbasis
    assert self._Nbasis <= Npoints, "Number of requested basis elements is larger than size of training set."
    
    # Allocate memory for greedy algorithm arrays
    dtype = type(np.asarray(training_space).flatten()[0])
    self.malloc(self._Nbasis, Npoints, Nsamples, Nmodes=Nmodes, dtype=dtype)
    
    # Seed 
    if Nbasis > 0:
      if self._loss == 'L2':
        self.errors[0] = np.max(self._norms)**2
      elif self._loss == 'Linfty':
        self.errors[0] = self.inner.Linfty(training_space[seed])
      self.indices[0] = seed
      self.basis[0] = training_space[seed]/self._norms[seed]
      self.basisnorms[0] = self._norms[seed]
      self.alpha[0] = self.alpha_arr(self.basis[0], training_space)
      #training_space[self.indices[0]] = np.ma.masked
  
  def iter(self, step, errs, training_space):
    """One iteration of standard reduced basis greedy algorithm.
    
    Updates the next entries of the errors, indices, basis, and 
    alpha arrays.
    
    Input
    -----
    step           -- current iteration step
    errs           -- projection errors across the training space
    training_space -- the training space of functions
    
    Examples
    --------
    
    If rb is an instance of StandardRB and iter=13 is the 13th 
    iteration of the greedy algorithm then the following code 
    snippet generates the next (i.e., 14th) entry of the errors, 
    indices, basis, and alpha arrays::
    
    >>> rb.iter(13)
    
    """
    
    next_index = np.argmax(errs)
    if next_index in self.indices:
      print(">>> Warning(Index already selected): Exiting greedy algorithm.")
      return 1
    else:
      self.indices[step+1] = np.argmax(errs)
      self.errors[step+1] = np.max(errs)
      self.basis[step+1], self.basisnorms[step+1] = self.add_basis(training_space[self.indices[step+1]], self.basis[:step+1])
      self.alpha[step+1] = self.alpha_arr(self.basis[step+1], training_space)
    #   #training_space[self.indices[step+1]] = np.ma.masked
    
    # self.indices[step+1] = np.argmax(errs)
    # self.errors[step+1] = np.max(errs)
    # self.basis[step+1], self.basisnorms[step+1] = self.add_basis(training_space[self.indices[step+1]], self.basis[:step+1])
    # self.alpha[step+1] = self.alpha_arr(self.basis[step+1], training_space)
    # training_space[self.indices[step+1]] = np.ma.masked
  
  def make(self, training_space, index_seed, tol, num=None, rel=False, verbose=False, timer=False):
    """Make a reduced basis using the standard greedy algorithm.
    
    Input
    -----
    training_space -- the training space of functions
    index_seed     -- array index for seed point in training set
    tol            -- tolerance that terminates the greedy algorithm
    rel            -- precomputed array of training set function norms 
                      (default is None)
    verbose        -- print projection errors to screen 
                      (default is False)
    timer          -- print elapsed time 
                      (default is False)
    
    Examples
    --------
    If rb is the StandardRB class instance, 0 the seed index, and
    T the training set then do::
     
    >>> rb.make(T, 0, 1e-12)
      
    To prevent displaying any print to screen, set the `verbose` 
    keyword argument to `False`::
    
    >>> rb.make(T, 0, 1e-12, verbose=False)
    
    """
    
    if num is None:
      self._Nbasis = len(training_space)
    else:
      assert type(num) is int, "Expecting integer."
      assert num >= 0, "Requested number of basis vectors must be non-negative."
      self._Nbasis = num
    
    # Seed the greedy algorithm
    self.seed(self._Nbasis, training_space, index_seed)
    
    # The standard greedy algorithm with fixed training set
    if verbose and self._Nbasis > 0:
      print("\nStep", "\t", "Error")
    if timer:
      t0 = time.time()
    
    if rel:
      #tol *= np.max(self._norms)**2
      tol *= self.errors[0]
    
    nn, flag = 0, 0
    while nn < self._Nbasis:
      if verbose:
        if rel:
          print(nn+1, "\t", self.errors[nn]/self.errors[0])
        else:
          print(nn+1, "\t", self.errors[nn])
        
      # Check if tolerance is met
      if self.errors[nn] <= tol:
        if nn == 0:
          nn += 1
        break
      # or if the number of basis vectors has been reached
      elif nn == self._Nbasis-1:
        nn += 1
        break
      # otherwise, add another point and basis vector
      else:
        # Single iteration and update errors, indices, basis, alpha arrays
        #errs = self.proj_errors_from_alpha(self.alpha[:nn+1], norms=self._norms)
        if self._loss == 'L2':
          errs = self.loss(self.alpha[:nn+1], norms=self._norms)
        elif self._loss == 'Linfty':
          errs = self.loss(self.alpha[:nn+1], self.basis[:nn+1], training_space)
        flag = self.iter(nn, errs, training_space)
      
      # If previously selected index is selected again then exit
      if flag == 1:
        nn += 1
        break
      # otherwise, increment the counter
      nn += 1
      
    if timer:
      print("\nElapsed time =", time.time()-t0)
      
    # Trim excess allocated entries
    self.size = nn
    self.trim(self.size)
  
  def project(self, f):
    """Project an array onto the reduced basis"""
    return self.projection_from_basis(f, self.basis)
  
  def trim(self, num):
    """Trim arrays to have size num"""
    self.errors = self.errors[:num]
    self.indices = self.indices[:num]
    self.basis = self.basis[:num]
    self.alpha = self.alpha[:num]
  
  def plot_errors(self, axis=None, 
                        plot='semilogy', 
                        rel=False,
                        color='k', 
                        marker='', 
                        linestyle='-', 
                        label=None, 
                        xlabel='Reduced basis size', 
                        ylabel='Max projection errors', 
                        show=True):
    """Plot maximum projection errors of training set as a function of basis size"""
    try:
      import matplotlib.pyplot as plt
    except ImportError:
      print("Cannot load matplotlib.pyplot module for plotting.")
    
    if hasattr(self, 'errors'):
      if axis is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
      else:
        ax = axis
      if rel:
        data = [list(range(1, self.size+1)), self.errors/self.errors[0]]
      else:
        data = [list(range(1, self.size+1)), self.errors]
      args = {'color': color, 
              'marker': marker,
              'linestyle': linestyle,
              'label': label,
            }
      lib.plot(plot, ax, data, args)
      ax.set_xlabel(xlabel)
      ax.set_ylabel(ylabel)
      
      if show:
        if label is not None:
          plt.legend(loc='lower left')
        plt.show()
    
    else:
      print("Object `errors` not found.")
    
    if ax is None:
      return fig, ax
    else:
      return ax



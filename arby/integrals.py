# --- integrals.py ---

"""    
    Classes and functions for computing inner products of functions
"""

import numpy as np
from lib import tuple_to_vstack, meshgrid


#########################
# Some helper functions #
#########################

def _rate_to_num(a, b, rate):
  """Convert sample rate to sample numbers in [a,b]"""
  return np.floor(np.float(b-a)*rate)+1

def _num_to_rate(a, b, num):
  """Convert sample numbers in [a,b] to sample rate"""
  return (num-1.)/np.float(b-a)

def _incr_to_num(a, b, incr):
  """Convert increment to sample numbers in [a,b]"""
  return _rate_to_num(a, b, 1./incr)

def _num_to_incr(a, b, num):
  """Convert sample numbers in [a,b] to increment"""
  return 1./_num_to_rate(a, b, num)

def grid(*a):
  """
  Make a meshgrid given a list of arrays.
  
  Input
  -----
  a -- list of arrays that will be formed in a grid
  
  Output
  ------
  Arrays of the meshgrid
  """
  return tuple_to_vstack(meshgrid(*a))

def _make_rules(interval, rule_dict, num=None, rate=None, incr=None):
  """The workhorse for making quadrature rules"""
  
  # Validate inputs
  input_dict = {'num':num, 'rate':rate, 'incr':incr}
  if list(input_dict.values()).count(None) != len(list(input_dict.keys()))-1:
    raise Exception("Must give input for only one of num, rate, or incr.")
  
  assert type(interval) in [list, np.ndarray], "List or array input required."
  len_interval = len(interval)
  
  # Extract and validate the sampling method requested
  for kk, vv in input_dict.items():
    if vv is not None:
      key = kk
      value = input_dict[kk]
      if type(value) in [list, np.ndarray]:
        len_arg = len(value)
      else:
        len_arg = 1
        value = [value]
  assert len_arg == len_interval-1, "Number of (sub)interval(s) does not equal number of arguments."
  
  # Generate nodes and weights for requested sampling
  nodes, weights = [], []
  for ii in range(len_arg):
    a, b = interval[ii:ii+2]
    n, w = rule_dict[key](a, b, value[ii])
    nodes.append(n)
    weights.append(w)
  
  return [np.hstack(nodes), np.hstack(weights)]
  
def _nodes_weights(interval=None, num=None, rate=None, incr=None, rule=None):
  """Wrapper to make nodes and weights for integration classes"""

  # Validate inputs
  assert interval, "Input to `interval` must not be None."
  values = [num, rate, incr]
  if values.count(None) != 2:
    raise Exception("Must give input for only one of num, rate, or incr.")
  if type(rule) is not str:
    raise Exception("Input to `rule` must be a string.")
  
  # Generate requested quadrature rule
  if rule in ['riemann', 'trapezoidal']:
    all_nodes, all_weights = QuadratureRules()[rule](interval, num=num, rate=rate, incr=incr)
  elif rule in ['chebyshev', 'chebyshev-lobatto', 'legendre', 'legendre-lobatto']:
    all_nodes, all_weights = QuadratureRules()[rule](interval, num=num)
  else:
    raise Exception("Requested quadrature rule (`%s`) not available." % rule)

  return all_nodes, all_weights, rule


##############################
# Class for quadrature rules #
##############################

class QuadratureRules(object):
  """Class for generating quadrature rules"""
  
  def __init__(self):
    self._dict = {
      'riemann': self.riemann,
      'trapezoidal': self.trapezoidal,
      'chebyshev': self.chebyshev,
      'chebyshev-lobatto': self.chebyshev_lobatto,
      'legendre': self.legendre,
      'legendre-lobatto': self.legendre_lobatto,
      }
    self.rules = list(self._dict.keys())
  
  def __getitem__(self, rule):
    return self._dict[rule]
  
  def riemann(self, interval, num=None, rate=None, incr=None):
    """
    Uniformly sampled array using Riemann quadrature rule
    over interval [a,b] with given sample number, sample rate
    or increment between samples.

    Input
    -----
    interval -- list indicating interval(s) for quadrature

    Options (specify only one)
    -------
    num  -- number(s) of quadrature points
    rate -- rate(s) at which points are sampled
    incr -- spacing(s) between samples

    Output
    ------
    nodes   -- quadrature nodes
    weights -- quadrature weights
    """

    rule_dict = {'num': self._riemann_num, 'rate': self._riemann_rate, 'incr': self._riemann_incr}
    return _make_rules(interval, rule_dict, num=num, rate=rate, incr=incr)

  def _riemann_num(self, a, b, n):
    """
    Uniformly sampled array using Riemann quadrature rule
    over given interval with given number of samples

    Input
    -----
    a -- start of interval
    b -- end of interval
    n -- number of quadrature points

    Output
    ------
    nodes   -- quadrature nodes
    weights -- quadrature weights
    """
    nodes = np.linspace( a, b, num=n )
    weights = np.ones(n, dtype='double')
    return [nodes, (b-a)/(n-1.)*weights]

  def _riemann_rate(self, a, b, rate):
    """
    Uniformly sampled array using Riemann quadrature rule
    over given interval with given sample rate

    Input
    -----
    a    -- start of interval
    b    -- end of interval
    rate -- sample rate

    Output
    ------
    nodes   -- quadrature nodes
    weights -- quadrature weights
    """
    # TODO: Check the assertion on 1/rate
    assert 1./rate <= abs(b-a), "Sample spacing is larger than interval. Increase sample rate."
    n = _rate_to_num(a, b, rate)
    return self._riemann_num(a, b, n)

  def _riemann_incr(self, a, b, incr):
    """
    Uniformly sampled array using Riemann quadrature rule
    over given interval with given sample spacing

    Input
    -----
    a    -- start of interval
    b    -- end of interval
    incr -- sample spacing

    Output
    ------
    nodes   -- quadrature nodes
    weights -- quadrature weights
    """
    assert incr <= abs(b-a), "Sample spacing is larger than interval. Decrease increment."
    n = _incr_to_num(a, b, incr)
    return self._riemann_num(a, b, n)
    
  def trapezoidal(self, interval, num=None, rate=None, incr=None):
    """
    Uniformly sampled array using the trapezoidal quadrature rule
    over interval [a,b] with given sample number, sample rate
    or increment between samples.

    Input
    -----
    interval -- list indicating interval(s) for quadrature

    Options (specify only one)
    -------
    num  -- number(s) of quadrature points
    rate -- rate(s) at which points are sampled
    incr -- spacing(s) between samples

    Output
    ------
    nodes   -- quadrature nodes
    weights -- quadrature weights
    """
    rule_dict = {'num': self._trapezoidal_num, 'rate': self._trapezoidal_rate, 'incr': self._trapezoidal_incr}
    return _make_rules(interval, rule_dict, num=num, rate=rate, incr=incr)

  def _trapezoidal_num(self, a, b, n):
    """
    Uniformly sampled array using the trapezoidal quadrature rule
    over given interval with given number of samples

    Input
    -----
    a -- start of interval
    b -- end of interval
    n -- number of quadrature points

    Output
    ------
    nodes   -- quadrature nodes
    weights -- quadrature weights
    """
    nodes = np.linspace( a, b, num=n )
    weights = np.ones(n, dtype='double')
    weights[0] = 0.5
    weights[-1] = 0.5
    return [nodes, (b-a)/(n-1.)*weights]

  def _trapezoidal_rate(self, a, b, rate):
    """
    Uniformly sampled array using the trapezoidal quadrature rule
    over given interval with given sample rate

    Input
    -----
    a    -- start of interval
    b    -- end of interval
    rate -- sample rate

    Output
    ------
    nodes   -- quadrature nodes
    weights -- quadrature weights
    """
    # TODO: Check the assertion on 1/rate
    assert 1./rate <= abs(b-a), "Sample spacing is larger than interval. Increase sample rate."
    n = _rate_to_num(a, b, rate)
    return self._trapezoidal_num(a, b, n)

  def _trapezoidal_incr(self, a, b, incr):
    """
    Uniformly sampled array using the trapezoidal quadrature rule
    over given interval with given sample spacing

    Input
    -----
    a    -- start of interval
    b    -- end of interval
    incr -- sample spacing

    Output
    ------
    nodes   -- quadrature nodes
    weights -- quadrature weights
    """
    assert incr <= abs(b-a), "Sample spacing is larger than interval. Decrease increment."
    n = _incr_to_num(a, b, incr)
    return self._trapezoidal_num(a, b, n)
    
  def chebyshev(self, interval, num):
    """
    Uniformly sampled array using Chebyshev-Gauss quadrature rule  
    over given interval with given number of samples
    
    Input
    -----
    interval -- list indicating interval(s) for quadrature
    num  -- number of quadrature points
    
    Output
    ------
    nodes   -- quadrature nodes
    weights -- quadrature weights
    """
    rule_dict = {'num': self._chebyshev}
    return _make_rules(interval, rule_dict, num=num)
    
  def _chebyshev(self, a, b, n):
    # Compute nodes and weights
    num = int(n)-1.
    nodes = np.array([-np.cos(np.pi*(2.*ii+1.)/(2.*num+2.)) for ii in range(int(n))])
    weights = np.pi/(num+1.) * np.sqrt(1.-nodes**2)
    return [nodes*(b-a)/2.+(b+a)/2., weights*(b-a)/2.]
    
  def chebyshev_lobatto(self, interval, num):
    """
    Uniformly sampled array using Chebyshev-Gauss-Lobatto quadrature rule  
    over given interval with given number of samples
    
    Input
    -----
    interval -- list indicating interval(s) for quadrature
    num  -- number of quadrature points
    
    Output
    ------
    nodes   -- quadrature nodes
    weights -- quadrature weights
    """
    rule_dict = {'num': self._chebyshev_lobatto}
    return _make_rules(interval, rule_dict, num=num)
    
  def _chebyshev_lobatto(self, a, b, n):
    # Compute nodes and weights
    num = int(n)-1.
    nodes = np.array([-np.cos(np.pi*ii/num) for ii in range(int(n))])
    weights = np.pi/num * np.sqrt(1.-nodes**2.)
    weights[0] /= 2. 
    weights[-1] /= 2.
    return [nodes*(b-a)/2.+(b+a)/2., weights*(b-a)/2.]
    
  def legendre(self, interval, num):
    raise Exception("Legendre-Gauss quadrature rule is not yet implemented.")
  
  def _legendre(self, a, b, n):
    pass
  
  def legendre_lobatto(self, interval, num):
    raise Exception("Legendre-Gauss-Lobatto quadrature rule is not yet implemented.")
    
  def _legendre_lobatto(self, a, b, n):
    pass


###################################################
# Class for computing inner products of functions #
###################################################

class Integration(object):
  """Integrals for computing inner products and norms of functions"""
  
  def __init__(self, interval=None, num=None, rate=None, incr=None, rule='trapezoidal', nodes=None, weights=None):
    if nodes is None and weights is None:
      self.nodes, self.weights, self.rule = _nodes_weights(interval, num, rate, incr, rule)
      #self._interval = interval
    else:
      if num is not None or rate is not None or incr is not None:
        print("\n>>>Warning: Using given nodes and weights to build quadrature rule.")
      self.nodes, self.weights = nodes, weights
    
    self.integrals = [
      'integral',
      'dot',
      'norm',
      'normalize',
      'match',
      'mismatch',
      'L2',
      'Ln',
      'Linfty'
    ]
  
  def integral(self, f):
    """Integral of a function"""
    return np.dot(self.weights, f)
  
  def dot(self, f, g):
    """Dot product of two functions"""
    return np.dot(self.weights, f.conjugate()*g)
  
  def norm(self, f):
    """Norm of function"""
    return np.sqrt(np.dot(self.weights, f.conjugate()*f).real)
  
  def normalize(self, f):
    """Normalize a function"""
    return f / self.norm(f)
  
  def match(self, f, g):
    """Match integral"""
    f_normed = f / self.norm(f)
    g_normed = g / self.norm(g)
    return np.dot(self.weights, f_normed.conjugate()*g_normed).real
  
  def mismatch(self, f, g):
    """Mismatch integral (1-match)"""
    return 1.-self.match(f, g)
  
  def Linfty(self, f):
    """L-infinity norm"""
    return np.abs(f).max()
    
  def Ln(self, f, n):
    """L-n norm"""
    assert n > 0
    return (np.dot(self.weights, np.abs(f)**n))**(1./n)
    
  def L2(self, f):
    """L-2 norm"""
    return np.sqrt(np.dot(self.weights, f.conjugate()*f).real)
    
  def _test_monomial(self, n=0):
    """Test integration rule by integrating the monomial x**n"""
    ans = self.integral(self.nodes**n)
    # FIXME: a, b are not part of the quadrature nodes for the Chebyshev rule.
    a, b = self.nodes[0], self.nodes[-1]
    expd = (b**(n+1.)-a**(n+1.))/(n+1.)
    
    print("\nExpected value for integral =", expd)
    print("Computed value for integral =", ans)
    print("\nAbsolute difference =", expd-ans)
    print("Relative difference =", 1.-ans/expd)


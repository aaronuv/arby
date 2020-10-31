"""
Helper functions used by several modules.
"""

import numpy as np, h5py
#from scipy.misc import factorial


def partitions(n, K):
  """Split array with n samples into K (nearly) equal partitions"""
  assert n >= K, "Number of partitions must not exceed number of samples."
  return np.asarray( np.array_split(np.arange(n), K) )


def random_partitions(n, K):
  """Split array with n samples into K (nearly) equal partitions of non-overlapping random subsets"""
  assert n >= K, "Number of folds must not exceed number of samples."
  
  # Make array with unique random integers
  rand = np.random.choice(range(n), n, replace=False)
  
  # Split into K (nearly) equal partitions
  return [np.sort(rand[pp]) for pp in partitions(n, K)]


def malloc(dtype, *nums):
  """Allocate some memory with given dtype"""
  return np.zeros(tuple(nums), dtype=dtype)


def malloc_more(arr, num_more):
  """Allocate more memory to append to arr"""
  dim = len(arr.shape)
  if dim == 1:
    return np.hstack([arr, malloc(arr.dtype, num_more)])
  elif dim == 2:
    # Add num_extra rows to arr
    shape = arr.shape
    return np.vstack([arr, malloc(arr.dtype, num_more, shape[1])])
  else:
    raise Exception("Expected a vector or matrix.")


def trim(arr, num):
  return arr[:num]


def scale_ab_to_cd(x, c, d):
  """Scale [a,b] to [c,d]"""
  a = x[0]; b = x[-1]
  a, b, c, d = map(float, [a, b, c, d])
  return (d-c)/(b-a)*x - (a*d-b*c)/(b-a)


def scale_ab_to_01(x):
  """Scale [a,b] to [0,1]"""
  interval = scale_ab_to_cd(x, 0, 1)
  return np.abs(interval)


def scale_01_to_ab(x, a, b):
  """Scale [0,1] to [a,b]"""
  if np.allclose(float(x[0]), 0.) and np.allclose(float(x[-1]), 1.):
    return scale_ab_to_cd(np.abs(x), a, b)
  else:	
    raise Exception("Expected a [0,...,1] array")


def get_arg(a, a0):
  """Get argument at which a0 occurs in array a"""
  return abs(a-a0).argmin()


def map_intervals(x, a, b):
  """Map array x to interval [a,b]"""
  M = (b-a)/(np.max(x)-np.min(x))
  B = a-M*np.min(x)
  return M*x+B


#def choose(top, bottom):
#  """Combinatorial choose function"""
#  return factorial(top)/factorial(bottom)/factorial(top-bottom)


def plot(plot_type, ax, data, args):
  assert type(plot_type) is str, "Expecting string input for `plot_type`."
  
  if plot_type == 'plot':
    ax.plot(*data, **args)
  elif plot_type == 'semilogy':
    ax.semilogy(*data, **args)
  elif plot_type == 'semilogx':
    ax.semilogx(*data, **args)
  elif plot_type == 'loglog':
    ax.loglog(*data, **args)
  else:
    print("Plot type not recognized. Choose between plot, semilogy, semilogx, or loglog.")
  
  return ax


def tuple_to_vstack(arr):
  return np.vstack(list(map(np.ravel, tuple(arr))))


def meshgrid(*arrs):
  """Multi-dimensional version of numpy's meshgrid"""
  arrs = tuple(reversed(arrs))  
  lens = list(map(len, arrs))
  dim = len(arrs)
  sz = 1
  for s in lens:
    sz*=s
  
  ans = []    
  for i, arr in enumerate(arrs):
    slc = [1]*dim
    slc[i] = lens[i]
    arr2 = np.asarray(arr).reshape(slc)
    for j, sz in enumerate(lens):
      if j!=i:
        arr2 = arr2.repeat(sz, axis=j) 
    ans.append(arr2)
  
  return ans[::-1]


def meshgrid_stack(*arrs):
  Arrs = meshgrid(*arrs)
  return tuple_to_vstack(Arrs).T


def chars_to_string(chars):
  """Convert list of integer ordinals to string of characters"""
  return "".join(chr(cc) for cc in chars)


def string_to_chars(string):
  """Convert string to list of integer ordinals"""
  return [ord(cc) for cc in string]


def cases_to_chars(cases):
  return string_to_chars( '|'.join(list(cases)) )


def chars_to_cases(chars):
  return chars_to_string(chars).split('|')


def h5open(file, mode):
  """
  Basic function for opening HDF5 files in requested mode or
  pass through a file or group descriptor.
  
  Input
  -----
    file -- Filename (str) or file/group descriptor of an open
            HDF5 file.
    mode -- Open file with given filename in requested mode.
  
  Output
  ------
    fp     -- Object of file instance or group descriptor of open file
    isopen -- True if file was successfully opened
  
  This function is used in the data.py module for the Data
  and CompositeData classes for loading and writing data to
  HDF5 file format.
  """
  fp = None
  isopen = False
  
  # If file is str then file is a filename and open as such in req'd mode
  if type(file) is str:
    try:
      fp = h5py.File(file, mode)
      isopen = True
    except IOError:
      print("Could not open file {} in {} mode.".format(file, mode))
  
  # If file is a HDF5 File or Group descriptor then assign file to fp
  elif hasattr(file, '__class__'):
    if file.__class__ in [h5py._hl.files.File, h5py._hl.group.Group]:
      fp = file
      isopen = True
  
  # Complain that file type is not recognized
  else:
    raise Exception("{} not recognized.".format(file))
  
  return fp, isopen


def h5close(file):
  if file.__class__ is h5py._hl.files.File:
    file.close()
    isopen = False
  else:
    isopen = True
  return isopen
    # if hasattr(options, 'close'):
    #   if options['close']:
    #     file.close()

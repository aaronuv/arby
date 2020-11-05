# --- lib.py ---

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE


"""
Helper functions used by several modules.
"""

import numpy as np


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


def tuple_to_vstack(arr):
    return np.vstack(list(map(np.ravel, tuple(arr))))


def meshgrid(*arrs):
    """Multi-dimensional version of numpy's meshgrid"""
    arrs = tuple(reversed(arrs))
    lens = list(map(len, arrs))
    dim = len(arrs)
    sz = 1
    for s in lens:
        sz *= s

    ans = []
    for i, arr in enumerate(arrs):
        slc = [1] * dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j != i:
                arr2 = arr2.repeat(sz, axis=j)
        ans.append(arr2)

    return ans[::-1]


def meshgrid_stack(*arrs):
    Arrs = meshgrid(*arrs)
    return tuple_to_vstack(Arrs).T

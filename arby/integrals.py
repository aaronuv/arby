# --- integrals.py ---

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE


"""
Classes and functions for computing inner products of functions
"""

import numpy as np


#########################
# Some helper functions #
#########################


def _rate_to_num(a, b, rate):
    """Convert sample rate to sample numbers in [a,b]"""
    return np.floor(np.float(b - a) * rate) + 1


def _incr_to_num(a, b, incr):
    """Convert increment to sample numbers in [a,b]"""
    return _rate_to_num(a, b, 1.0 / incr)


def _make_rules(interval, rule_dict, num=None, rate=None, incr=None):
    """The workhorse for making quadrature rules"""

    # Validate inputs
    input_dict = {"num": num, "rate": rate, "incr": incr}

    assert type(interval) in [
        list,
        np.ndarray,
    ], "List or array input required."
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
    assert (
        len_arg == len_interval - 1
    ), "Number of (sub)interval(s) does not equal number of arguments."

    # Generate nodes and weights for requested sampling
    nodes, weights = [], []
    for ii in range(len_arg):
        a, b = interval[ii: ii + 2]
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
        raise ValueError("Must give input for only one of num, rate, or incr.")
    if type(rule) is not str:
        raise TypeError("Input `rule` must be a string.")

    # Generate requested quadrature rule
    if rule in ["riemann"]:
        all_nodes, all_weights = QuadratureRules()[rule](
            interval, num=num, rate=rate, incr=incr
        )
    else:
        raise ValueError(f"Requested quadrature rule ({rule}) not available.")
    return all_nodes, all_weights, rule


##############################
# Class for quadrature rules #
##############################


class QuadratureRules:
    """Class for generating quadrature rules"""

    def __init__(self):
        self._dict = {"riemann": self.riemann}
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

        rule_dict = {"num": self._riemann_num}
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
        nodes = np.linspace(a, b, num=n)
        weights = np.ones(n, dtype="double")
        weights[-1] = 0.0
        return [nodes, (b - a) / (n - 1.0) * weights]


class Integration:
    """Integrals for computing inner products and norms of functions"""

    def __init__(
        self,
        interval=None,
        num=None,
        rate=None,
        incr=None,
        rule="riemann",
        nodes=None,
        weights=None,
    ):

        self.nodes, self.weights, self.rule = _nodes_weights(
            interval, num, rate, incr, rule
        )

        self.integrals = ["integral", "dot", "norm", "normalize", "L2"]

    def integral(self, f):
        """Integral of a function"""
        return np.dot(self.weights, f)

    def dot(self, f, g):
        """Dot product between a function f and an array of functions g"""
        return np.dot(self.weights, (f.conjugate() * g).transpose())

    def norm(self, f):
        """Norm of function"""
        return np.sqrt(np.dot(self.weights, f.conjugate() * f).real)

    def normalize(self, f):
        """Normalize a function"""
        return f / self.norm(f)

    def L2(self, f):
        """L-2 norm"""
        return np.sqrt(np.dot(self.weights, f.conjugate() * f).real)

# --- integrals.py ---

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE


"""
Classes and functions to define an integration scheme
"""

import numpy as np


###################
# Helper functions
###################


def _rate_to_num(a, b, rate):
    """Convert sample rate to sample numbers in [a,b]."""
    return np.floor(np.float(b - a) * rate) + 1


def _incr_to_num(a, b, incr):
    """Convert increment to sample numbers in [a,b]."""
    return _rate_to_num(a, b, 1.0 / incr)


def _nodes_weights(interval=None, num=None, rule=None):
    """Wrapper to make nodes and weights for integration classes"""

    # Validate inputs
    if interval is None:
        raise ValueError("`interval` must be provided.")
    if type(rule) is not str:
        raise TypeError("Input `rule` must be a string.")

    # Generate requested quadrature rule
    if rule in ["riemann", "trapezoidal"]:
        all_nodes, all_weights = QuadratureRules()[rule](
            interval, num=num)
    else:
        raise ValueError(f"Requested quadrature rule ({rule}) not available.")
    return all_nodes, all_weights, rule


##############################
# Class for quadrature rules #
##############################


class QuadratureRules:
    """Class for generating quadrature rules"""

    def __init__(self):
        self._dict = {"riemann": self.riemann, "trapezoidal": self.trapezoidal}
        self.rules = list(self._dict.keys())

    def __getitem__(self, rule):
        return self._dict[rule]

    def riemann(self, interval, num=None):
        """
        Uniformly sampled array using Riemann quadrature rule over interval
        [a,b] with given sample number, sample rate or increment between
        samples.

        Parameters
        ----------
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

        nodes = np.linspace(a, b, num=n)
        weights = np.ones(n, dtype="double")
        weights[-1] = 0.0
        return [nodes, (b - a) / (n - 1) * weights]

        return _make_rules(interval, rule_dict, num=num)

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


    def trapezoidal(self, interval, num=None, rate=None, incr=None):
        """ Trapezoidal rule."""

        rule_dict = {"num": self._trapezoidal_num}
        return _make_rules(interval, rule_dict, num=num, rate=rate, incr=incr)

    def _trapezoidal_num(self, a, b, n):
        nodes = np.linspace(a, b, num=n)
        weights = np.ones(n, dtype="double")
        weights[0] = 0.5
        weights[-1] = 0.5
        return [nodes, (b - a) / (n - 1) * weights]



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

        self.nodes, self.weights, self.rule = _nodes_weights(interval, num,
                                                             rule)

        self.integrals = ["integral", "dot", "norm", "normalize"]

    def integral(self, f):
        """Integral of a function."""
        return np.dot(self.weights, f)

    def dot(self, f, g):
        """Dot product between a function f and an array of functions g."""
        return np.dot(self.weights, (f.conjugate() * g).transpose())

    def norm(self, f):
        """Norm of function."""
        f_euclid = (f.conjugate() * f).transpose().real
        return np.sqrt(np.dot(self.weights, f_euclid))

    def normalize(self, f):
        """Normalize a function."""
        return f / self.norm(f)

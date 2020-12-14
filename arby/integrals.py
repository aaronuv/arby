# --- integrals.py ---

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE


"""Classes and functions to define integration schemes."""

import numpy as np


###################
# Helper functions
###################


def _nodes_weights(interval=None, rule=None):
    """Build nodes and weights."""
    # Validate inputs
    if interval is None:
        raise ValueError("`interval` must be provided.")
    if type(rule) is not str:
        raise TypeError("Input `rule` must be a string.")

    # Generate requested quadrature rule
    if rule in ["riemann", "trapezoidal"]:
        all_nodes, all_weights = Quadratures()[rule](interval)
    else:
        raise ValueError(f"Requested quadrature rule ({rule}) not available.")
    return all_nodes, all_weights, rule


##############################
# Class for quadrature rules #
##############################


class Quadratures:
    """Quadrature rules class."""

    def __init__(self):
        self._dict = {
            "riemann": self._riemann,
            "trapezoidal": self._trapezoidal,
        }

    def __getitem__(self, rule):
        """Get the quadrature rule."""
        return self._dict[rule]

    def _riemann(self, interval):
        """Uniform Riemann quadrature.

        Parameters
        ----------
        interval: numpy.ndarray
            The set of points on which define the quadrature.

        Returns
        -------
        nodes: numpy.ndarray
            Quadrature nodes.
        weights: numpy.ndarray
            Quadrature weights.

        """
        n = interval.shape[0]
        a = interval.min()
        b = interval.max()
        weights = np.ones(n, dtype="double")
        weights[-1] = 0.0
        nodes = interval
        return [nodes, (b - a) / (n - 1) * weights]

    def _trapezoidal(self, interval):
        """Uniform trapezoidal quadrature."""
        n = interval.shape[0]
        a = interval.min()
        b = interval.max()
        weights = np.ones(n, dtype="double")
        weights[0] = 0.5
        weights[-1] = 0.5
        nodes = interval
        return [nodes, (b - a) / (n - 1) * weights]


class Integration:
    """Comprise an integration scheme.

    Parameters
    ----------
    interval: numpy.ndarray
        Set of points to be used for integrals.
    rule: str, optional
        Quadrature rule. Default = "riemann".

    """

    def __init__(self, interval, rule="riemann"):

        self.nodes, self.weights, self.rule = _nodes_weights(interval, rule)

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

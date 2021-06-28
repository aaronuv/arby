# --- integrals.py ---

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Integration schemes module."""

# =============================================================================
# IMPORTS
# =============================================================================

import numba

import numpy as np


# =============================================================================
# Helper functions
# =============================================================================


@numba.njit(parallel=True)
def _riemann_quadrature(interval):
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
    return nodes, (b - a) / (n - 1) * weights


@numba.njit(parallel=True)
def _trapezoidal_quadrature(interval):
    """Uniform trapezoidal quadrature."""
    n = interval.shape[0]
    a = interval.min()
    b = interval.max()
    weights = np.ones(n, dtype="double")
    weights[0] = 0.5
    weights[-1] = 0.5
    nodes = interval
    return nodes, (b - a) / (n - 1) * weights


@numba.njit(parallel=True)
def _euclidean_quadrature(interval):
    """Uniform euclidean quadrature.

    This quadrature provides discrete inner products for intrinsically discrete
    data.

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
    weights = np.ones(n, dtype="double")
    nodes = interval
    return nodes, weights


QUADRATURES = {
    "riemann": _riemann_quadrature,
    "trapezoidal": _trapezoidal_quadrature,
    "euclidean": _euclidean_quadrature,
}


# =============================================================================
# Class for quadrature rules
# =============================================================================


INTEGRATION_CLASS_SPEC = [
    ("interval", numba.types.float64[:]),
    ("rule", numba.types.string),
    ("nodes_", numba.types.float64[:]),
    ("weights_", numba.types.float64[:]),
]


@numba.experimental.jitclass(INTEGRATION_CLASS_SPEC)
class Integration:
    """Integration scheme.

    This class fixes a frame for performing integrals, inner products and
    derived operations. An integral is defined by a quadrature rule composed
    by nodes and weights which are used to construct a discrete approximation
    to the true integral (or inner product).

    For completeness, an "euclidean" rule is available for which inner products
    reduce to simple discrete dot products.

    Parameters
    ----------
    interval : numpy.ndarray
        Equispaced set of points as domain for integrals or inner products.
    rule : str, optional
        Quadrature rule. Default = "riemann". Available = ("riemann",
        "trapezoidal", "euclidean")

    """

    def __init__(self, interval, rule="riemann"):

        self.interval = interval
        self.rule = rule

        quadrature_result = self._quadrature(interval)
        if quadrature_result is None:
            raise ValueError("Unknow rule provided")
        self.nodes_, self.weights_ = quadrature_result

    def _quadrature(self, interval):
        if self.rule == "riemann":
            return _riemann_quadrature(interval)
        elif self.rule == "trapezoidal":
            return _trapezoidal_quadrature(interval)
        elif self.rule == "euclidean":
            return _euclidean_quadrature(interval)
        return None

    def integral(self, f):
        """Integrate a function.

        Parameters
        ----------
        f : np.ndarray
            Real or complex numbers array.

        """
        f_ = np.ascontiguousarray(f)
        w_ = np.ascontiguousarray(self.weights_)
        return np.dot(w_, f_)

    def dot(self, f, g):
        """Return the dot product between functions.

        Parameters
        ----------
        f, g : np.ndarray
            Real or complex numbers array.

        """
        fgT = (f.conjugate() * g).transpose()
        weights = np.asarray(self.weights_, dtype=fgT.dtype)
        f_ = np.ascontiguousarray(fgT)
        w_ = np.ascontiguousarray(weights)
        return np.dot(w_, f_)

    def norm(self, f):
        """Return the norm of a function.

        Parameters
        ----------
        f : np.ndarray
            Real or complex numbers array.

        """
        f_euclid = (f.conjugate() * f).transpose().real
        f_ = np.ascontiguousarray(f_euclid)
        w_ = np.ascontiguousarray(self.weights_)
        return np.sqrt(np.dot(w_, f_))

    def normalize(self, f):
        """Normalize a function.

        Parameters
        ----------
        f : np.ndarray
            Real or complex numbers array.

        """
        normf = np.asarray(self.norm(f))
        return np.divide(f, normf.reshape(-1, 1))

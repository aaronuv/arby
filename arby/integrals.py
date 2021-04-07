# --- integrals.py ---

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Classes and functions to define integration schemes."""

# =============================================================================
# IMPORTS
# =============================================================================

import attr

import numpy as np

# =============================================================================
# Helper functions
# =============================================================================


def riemann_quadrature(interval):
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


def trapezoidal_quadrature(interval):
    """Uniform trapezoidal quadrature."""
    n = interval.shape[0]
    a = interval.min()
    b = interval.max()
    weights = np.ones(n, dtype="double")
    weights[0] = 0.5
    weights[-1] = 0.5
    nodes = interval
    return nodes, (b - a) / (n - 1) * weights


def euclidean_quadrature(interval):
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
    "riemann": riemann_quadrature,
    "trapezoidal": trapezoidal_quadrature,
    "euclidean": euclidean_quadrature,
}


# =============================================================================
# Class for quadrature rules
# =============================================================================


@attr.s(frozen=True)
class Integration:
    """Comprises an integration scheme.

    This class defines a scheme to perform integrals, inner products and
    derived operations. An integral is defined by a quadrature rule composed
    by nodes and weights used to construct a discrete approximation to the true
    integral (or inner product).

    For completeness, an "euclidean" rule is available for which inner products
    reduce to simple discrete dot products.

    Parameters
    ----------
    interval : numpy.ndarray
        Set of points to be used for integrals or inner products.
    rule : str, optional
        Quadrature rule. Default = "riemann". Available = ("riemann",
        "trapezoidal", "euclidean")

    """

    interval = attr.ib()
    rule = attr.ib(
        validator=attr.validators.in_(QUADRATURES), default="riemann"
    )

    nodes_ = attr.ib(init=False, repr=False)
    weights_ = attr.ib(init=False, repr=False)

    def __attrs_post_init__(self):  # noqa to skip pydocstyle in the method
        quadrature = QUADRATURES[self.rule]
        nodes, weights = quadrature(self.interval)

        super().__setattr__("nodes_", nodes)
        super().__setattr__("weights_", weights)

    def integral(self, f):
        """Integral of a function.

        Parameters
        ----------
        f : np.ndarray
            Real or complex numbers array.

        """
        return np.dot(self.weights_, f)

    def dot(self, f, g):
        """Dot product between functions f and g.

        Parameters
        ----------
        f, g : np.ndarray
            Real or complex numbers array.

        """
        return np.dot(self.weights_, (f.conjugate() * g).transpose())

    def norm(self, f):
        """Norm of a function.

        Parameters
        ----------
        f : np.ndarray
            Real or complex numbers array.

        """
        f_euclid = (f.conjugate() * f).transpose().real
        return np.sqrt(np.dot(self.weights_, f_euclid))

    def normalize(self, f):
        """Normalize a function.

        Parameters
        ----------
        f : np.ndarray
            Real or complex numbers array.

        """
        return f / self.norm(f)

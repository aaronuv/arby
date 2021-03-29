# - rom.py -

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE

"""Basis analysis class."""

import functools
import logging
from collections import namedtuple

import attr

import numpy as np

from .integrals import Integration

# ================
# Constants
# ================

if hasattr(functools, "cached_property"):

    cached_property = functools.cached_property

else:

    def cached_property(func):  # pragma: no cover
        """Workaround for functools.cached_property for Python < 3.8."""
        cache = functools.lru_cache(maxsize=None)
        cached_func = cache(func)
        cached_prop = property(cached_func)
        return cached_prop


logger = logging.getLogger("arby.basis")


# =================================
# Class for Basis Analysis
# =================================

#: Simple container of the Empirical Interpolantion Matrix
EIM = namedtuple("EIM", ["interpolant", "nodes"])


@attr.s(frozen=True, hash=False)
class Basis:
    """Basis elements utilities.

    Parameters
    ----------
    data : numpy.ndarray
        The reduced basis of the Reduced Order Model.
    integration: arby.integrals.Integration
        Instance of the `Integration` class.

    Attributes
    ----------
    Nbasis_: int
        Number of basis elements.


    References
    ----------
    .. [field2014fast] Scott E. Field, Chad R. Galley, Jan S. Hesthaven,
        Jason Kaye, and Manuel Tiglio. Fast Prediction and Evaluation of
        Gravitational Waveforms Using Surrogate Models. Phys. Rev. X 4, 031006

    """

    data: np.ndarray = attr.ib(converter=np.asarray)
    integration: np.ndarray = attr.ib(
        validator=attr.validators.instance_of(Integration)
    )
    Nbasis_: int = attr.ib(init=False)

    # ==== Attrs orchestration=================================================

    @Nbasis_.default
    def _Nbasis__default(self):
        return self.data.shape[0]

    # ====== Empirical Interpolation Method ===================================

    def _next_vandermonde(self, data, nodes, vandermonde=None):
        """Build the next Vandermonde matrix from the previous one."""
        if vandermonde is None:
            vandermonde = [[data[0, nodes[0]]]]
            return vandermonde

        n = len(vandermonde)
        new_node = nodes[-1]
        for i in range(n):
            vandermonde[i].append(data[i, new_node])

        vertical_vector = [data[n, nodes[j]] for j in range(n)]
        vertical_vector.append(data[n, new_node])
        vandermonde.append(vertical_vector)
        return vandermonde

    @cached_property
    def eim_(self):
        """Empirical Interpolantion matrix.

        Implement the Empirical Interpolation Method [field2014fast]_ to select
        a set of interpolation nodes from the physical interval and build an
        interpolant matrix.

        Raises
        ------
        ValueError
            If there is no basis for EIM.

        """
        nodes = []
        v_matrix = None
        first_node = np.argmax(np.abs(self.data[0]))
        nodes.append(first_node)

        logger.debug(first_node)

        for i in range(1, self.Nbasis_):
            v_matrix = self._next_vandermonde(self.data, nodes, v_matrix)
            base_at_nodes = [self.data[i, t] for t in nodes]
            invV_matrix = np.linalg.inv(v_matrix)
            step_basis = self.data[:i]
            basis_interpolant = base_at_nodes @ invV_matrix @ step_basis
            residual = self.data[i] - basis_interpolant
            new_node = np.argmax(abs(residual))

            logger.debug(new_node)

            nodes.append(new_node)

        v_matrix = np.array(self._next_vandermonde(self.data, nodes, v_matrix))
        invV_matrix = np.linalg.inv(v_matrix.transpose())
        interpolant = self.data.transpose() @ invV_matrix

        return EIM(interpolant=interpolant, nodes=nodes)

    def projection_error(self, h):
        """Square of the projection error of a function onto a basis.

        The error is computed in the L2 norm.

        Parameters
        ----------
        h: numpy.array
            Function or set of functions to be projected.
        basis: numpy.array
            Orthonormal basis.

        Returns
        -------
        l2_error: float or numpy.array
            Square of the projection error.
        """
        h_norm = self.integration.norm(h).real
        inner_prod = np.array(
            [self.integration.dot(basis_elem, h) for basis_elem in self.data]
        )
        l2_error = h_norm ** 2 - np.linalg.norm(inner_prod) ** 2
        return l2_error

    def project(self, h):
        """Project a function h on the basis.

        Parameters
        ----------
        h: numpy.array
            Function or set of functions to be projected.

        Returns
        -------
        projected_function: numpy.array
            Projection of h on the given basis.
        """
        projected_function = 0.0
        for e in self.data:
            projected_function += e * self.integration.dot(e, h)
        return projected_function

    def interpolate(self, h):
        """Interpolate a function h at EIM nodes.

        Parameters
        ----------
        h: numpy.array
            Function or set of functions to be interpolated.

        Returns
        -------
        h_interpolated: numpy.array
            Function h interpolated at EIM nodes.
        """
        h_at_nodes = np.array([h[eim_node] for eim_node in self.eim_.nodes])
        h_interpolated = self.eim_.interpolant @ h_at_nodes
        return h_interpolated

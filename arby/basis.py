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


# =============================================================================
# FUNCTIONS
# =============================================================================


def _gs_one_element(h, basis, integration, max_iter=3):  # pragma: no cover
    """Orthonormalize a function against an orthonormal basis."""
    norm = integration.norm(h)
    e = h / norm

    n_iters = 0
    continue_loop = True
    while continue_loop:
        for b in basis:
            e -= b * integration.dot(b, e)
        new_norm = integration.norm(e)

        a = 0.5
        if new_norm / norm <= a:
            norm = new_norm
            n_iters += 1
            if n_iters > max_iter:
                raise StopIteration(
                    "Gram-Schmidt algorithm: max number of "
                    "iterations reached ({}).".format(max_iter)
                )
        else:
            continue_loop = False

    return e / new_norm, new_norm


def _sq_prog_errors(proj_matrix, norms, Ntrain):  # pragma: no cover
    """Square of projection errors.

    Parameters
    ----------
    proj_matrix : numpy.ndarray, shape=(n,`Ntrain`)
        Stores the projection coefficients of the training functions. n
        is the number of basis elements.
    norms : numpy.ndarray, shape=(`Ntrain`)
        Stores the norms of the training functions.

    Returns
    -------
    proj_errors : numpy.ndarray, shape=(`Ntrain`)
        Squared projection errors.
    """
    proj_norms = np.array(
        [np.linalg.norm(proj_matrix[:, i]) for i in range(Ntrain)]
    )
    proj_errors = norms ** 2 - proj_norms ** 2
    return proj_errors


def _prune(greedy_errors, proj_matrix, num):  # pragma: no cover
    """Prune arrays to have size num."""
    return greedy_errors[:num], proj_matrix[:num]


def reduce_basis(
    training_space,
    physical_interval,
    parameter_interval,
    integration_rule="riemann",
    greedy_tol=1e-12,
) -> tuple:  # pragma: no cover
    """Reduced Basis greedy algorithm implementation.

    Algorithm  to build an orthonormal basis from training data. This
    basis reproduces the training functions by means of projection within a
    tolerance specified by the user [field2014fast]_.

    Returns
    -------
    basis : arby.Basis
        The reduced basis of the Reduced Order Model.
    greedy_error: np.ndarray.
        Error of the greedy algorithm.

    Raises
    ------
    ValueError
        If ``training_space.shape[1]`` doesn't coincide with weights of the
        quadrature rule.

    """
    integration = Integration(physical_interval, rule=integration_rule)

    # useful information
    Ntrain = training_space.shape[0]
    Nsamples = training_space.shape[1]

    # Validate inputs
    if Nsamples != np.size(integration.weights_):
        raise ValueError(
            "Number of samples is inconsistent with quadrature rule."
        )

    # If seed gives a null function, choose a random seed
    index_seed = 0
    seed_function = training_space[index_seed]
    zero_function = np.zeros_like(seed_function)
    while np.allclose(seed_function, zero_function):
        index_seed = np.random.randint(1, Ntrain)
        seed_function = training_space[index_seed]

    # ====== Seed the greedy algorithm and allocate memory ======

    # Allocate memory for greedy algorithm arrays
    greedy_errors = np.empty(Ntrain, dtype="double")
    basisnorms = np.empty(Ntrain, dtype="double")
    proj_matrix = np.empty((Ntrain, Ntrain), dtype=training_space.dtype)

    norms = integration.norm(training_space)

    # Seed
    greedy_indices = [index_seed]
    basis_data = np.empty_like(training_space)
    basis_data[0] = training_space[index_seed] / norms[index_seed]

    basisnorms[0] = norms[index_seed]
    proj_matrix[0] = integration.dot(basis_data[0], training_space)

    errs = _sq_prog_errors(proj_matrix[:1], norms=norms, Ntrain=Ntrain)
    next_index = np.argmax(errs)
    greedy_errors[0] = errs[next_index]
    sigma = greedy_errors[0]

    # ====== Start greedy loop ======
    logger.debug("\n Step", "\t", "Error")
    nn = 0
    while sigma > greedy_tol:
        nn += 1

        if next_index in greedy_indices:

            # Prune excess allocated entries
            greedy_errors, proj_matrix = _prune(greedy_errors, proj_matrix, nn)
            return (
                Basis(data=basis_data[:nn], integration=integration),
                greedy_errors,
            )

        greedy_indices.append(next_index)
        basis_data[nn], basisnorms[nn] = _gs_one_element(
            training_space[greedy_indices[nn]],
            basis_data[:nn],
            integration,
        )
        proj_matrix[nn] = integration.dot(basis_data[nn], training_space)
        errs = _sq_prog_errors(
            proj_matrix[: nn + 1], norms=norms, Ntrain=Ntrain
        )
        next_index = np.argmax(errs)
        greedy_errors[nn] = errs[next_index]

        sigma = errs[next_index]

        logger.debug(nn, "\t", sigma)

    # Prune excess allocated entries
    greedy_errors, proj_matrix = _prune(greedy_errors, proj_matrix, nn + 1)

    return (
        Basis(data=basis_data[: nn + 1], integration=integration),
        greedy_errors,
    )

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

from . import integrals


# =================================
# CONSTANTS
# =================================


logger = logging.getLogger("arby.basis")


# =================================
# Class for Basis Analysis
# =================================

# Container for EIM information
EIM = namedtuple("EIM", ["interpolant", "nodes"])

# Container for RB information
RB = namedtuple("RB", ["basis", "indices", "errors", "projection_matrix"])


@attr.s(frozen=True, hash=False)
class Basis:
    """Basis elements utilities.

    Parameters
    ----------
    data : numpy.ndarray
        The reduced basis of the Reduced Order Model.
    integration: arby.integrals.Integration
        Instance of the `Integration` class.


    References
    ----------
    .. [field2014fast] Scott E. Field, Chad R. Galley, Jan S. Hesthaven,
        Jason Kaye, and Manuel Tiglio. Fast Prediction and Evaluation of
        Gravitational Waveforms Using Surrogate Models. Phys. Rev. X 4, 031006

    """

    data: np.ndarray = attr.ib(converter=np.asarray)
    integration: np.ndarray = attr.ib(
        validator=attr.validators.instance_of(integrals.Integration)
    )

    # ==== Attrs orchestration=================================================

    @property
    def Nbasis_(self) -> int:
        """Return the number of basis elements."""
        return self.data.shape[0]

    @property
    def size_(self) -> int:
        """Return the total number of elements in the basis."""
        return self.data.size

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

    @property
    @functools.lru_cache(maxsize=None)
    def eim(self) -> EIM:
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

        return EIM(interpolant=interpolant, nodes=tuple(nodes))

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
        h_at_nodes = np.array([h[eim_node] for eim_node in self.eim.nodes])
        h_interpolated = self.eim.interpolant @ h_at_nodes
        return h_interpolated


# =============================================================================
# INTERNAL FUNCTIONS
# =============================================================================


def _gs_one_element(h, basis, integration, max_iter=3):
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


def _sq_proj_errors(proj_matrix, norms, Ntrain):
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


def _prune(greedy_errors, proj_matrix, num):
    """Prune arrays to have size num."""
    return greedy_errors[:num], proj_matrix[:num]


# =============================================================================
# PUBLIC FUNCTIONS
# =============================================================================

# ===================================
#    Iterated-Modified Gram-Schmidt
#      Orthonormalization Function
# ===================================


def gram_schmidt(functions, integration, max_iter=3) -> np.ndarray:
    """Orthonormalize a set of functions.

    This algorithm implements the Iterated, Modified Gram-Schmidt (GS)
    algorithm to build an orthonormal basis from a set of functions
    described in [hoffmann1989iterative]_.

    Parameters
    ----------
    functions : array_like, shape=(m, L)
        Functions to be orthonormalized, where m is the number of functions
        and L is the sample length.
    integration : arby.integrals.Integration
        Instance of the `Integration` class.
    max_iter : int, optional
        Maximum number of interations. Default = 3.

    Returns
    -------
    basis : numpy.ndarray
        Orthonormalized array.

    Raises
    ------
    ValueError
        If functions are not linearly independent.

    References
    ----------
    .. [hoffmann1989iterative] Hoffmann, W. Iterative algorithms for
      Gram-Schmidt orthogonalization. Computing 41, 335-348 (1989).
      https://doi.org/10.1007/BF02241222

    """
    functions = np.asarray(functions)

    _, svds, _ = np.linalg.svd(functions)

    linear_indep_tol = 5e-15
    if np.min(svds) < linear_indep_tol:
        raise ValueError("Functions are not linearly independent.")

    ortho_basis = []

    # First element of the basis is special, it's just normalized
    ortho_basis.append(integration.normalize(functions[0]))

    # For the rest of basis elements add them one by one by extending basis
    for new_basis_elem in functions[1:]:
        projected_element, _ = _gs_one_element(
            new_basis_elem, ortho_basis, integration, max_iter
        )
        ortho_basis.append(projected_element)
    basis_data = np.array(ortho_basis)

    return basis_data


def reduced_basis(
    training_set,
    physical_points,
    integration_rule="riemann",
    greedy_tol=1e-12,
) -> tuple:
    """Reduce Basis greedy algorithm implementation.

    Algorithm  to build an orthonormal basis from training data. This
    basis reproduces the training functions by means of projection within a
    tolerance specified by the user [field2014fast]_.

    Returns
    -------
    basis_ : arby.Basis
        The reduced basis of the Reduced Order Model.
    greedy_errors_: np.ndarray.
        Error of the greedy algorithm.
    projection_matrix_: np.ndarray.
        Projection coefficients from the greedy algorithm.

    Raises
    ------
    ValueError
        If ``training_set.shape[1]`` doesn't coincide with weights of the
        quadrature rule.

    """
    integration = integrals.Integration(physical_points, rule=integration_rule)

    # useful information
    Ntrain = training_set.shape[0]
    Nsamples = training_set.shape[1]

    # Validate inputs
    if Nsamples != np.size(integration.weights_):
        raise ValueError(
            "Number of samples is inconsistent with quadrature rule."
        )

    # If seed gives a null function, choose a random seed
    index_seed = 0
    seed_function = training_set[index_seed]
    zero_function = np.zeros_like(seed_function)
    while np.allclose(seed_function, zero_function):
        index_seed = np.random.randint(1, Ntrain)
        seed_function = training_set[index_seed]

    # ====== Seed the greedy algorithm and allocate memory ======

    # Allocate memory for greedy algorithm arrays
    greedy_errors = np.empty(Ntrain, dtype="double")
    basisnorms = np.empty(Ntrain, dtype="double")
    proj_matrix = np.empty((Ntrain, Ntrain), dtype=training_set.dtype)

    norms = integration.norm(training_set)

    # Seed
    greedy_indices = [index_seed]
    basis_data = np.empty_like(training_set)
    basis_data[0] = training_set[index_seed] / norms[index_seed]

    basisnorms[0] = norms[index_seed]
    proj_matrix[0] = integration.dot(basis_data[0], training_set)

    errs = _sq_proj_errors(proj_matrix[:1], norms=norms, Ntrain=Ntrain)
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
            return RB(
                basis=Basis(
                    data=basis_data[: nn + 1], integration=integration
                ),
                indices=greedy_indices,
                errors=greedy_errors,
                projection_matrix=proj_matrix,
            )

        greedy_indices.append(next_index)
        basis_data[nn], basisnorms[nn] = _gs_one_element(
            training_set[greedy_indices[nn]],
            basis_data[:nn],
            integration,
        )
        proj_matrix[nn] = integration.dot(basis_data[nn], training_set)
        errs = _sq_proj_errors(
            proj_matrix[: nn + 1], norms=norms, Ntrain=Ntrain
        )
        next_index = np.argmax(errs)
        greedy_errors[nn] = errs[next_index]

        sigma = errs[next_index]

        logger.debug(nn, "\t", sigma)

    # Prune excess allocated entries
    greedy_errors, proj_matrix = _prune(greedy_errors, proj_matrix, nn + 1)

    return RB(
        basis=Basis(data=basis_data[: nn + 1], integration=integration),
        indices=tuple(greedy_indices),
        errors=greedy_errors,
        projection_matrix=proj_matrix,
    )

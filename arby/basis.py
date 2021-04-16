# - basis.py -

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE

"""Basis analysis module."""

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

#: Container for EIM information
EIM = namedtuple("EIM", ["interpolant", "nodes"])
EIM.interpolant.__doc__ = "Interpolant matrix."
EIM.nodes.__doc__ = "EIM nodes."

#: Container for RB information
RB = namedtuple("RB", ["basis", "indices", "errors", "projection_matrix"])
RB.basis.__doc__ = "Reduced basis object."
RB.indices.__doc__ = "Greedy indices."
RB.errors.__doc__ = "Greedy errors."
RB.projection_matrix.__doc__ = "Projection coefficients."


@attr.s(frozen=True, hash=False)
class Basis:
    """Basis object and utilities.

    Create a basis object introducing an orthonormalized set of functions
    ``data`` and an ``integration`` class instance to enable integration
    utilities for the basis.

    Parameters
    ----------
    data : numpy.ndarray
        Orthonormalized basis.
    integration : arby.integrals.Integration
        Instance of the ``Integration`` class.

    -->
    Attributes
    ----------
    Nbasis_ : int
        Number of basis elements.
    eim_ : arby.basis.EIM
        Container storing EIM information: ``Interpolant`` matrix and EIM
        ``nodes``.

    Methods
    -------
    interpolate(h)
        Interpolate a function h at EIM nodes.
    project(h)
        Project a function h onto the basis.
    projection_error(h)
        Compute the error from the projection of h onto the basis.
    -->

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
    def eim_(self) -> EIM:
        """Implement EIM algorithm.

        The Empirical Interpolation Method (EIM) [TiglioAndVillanueva2021]_
        introspects the basis and selects a set of interpolation ``nodes`` from
        the physical domain for building an ``interpolant`` matrix using the
        basis and the selected nodes. The ``interpolant`` matrix can be used to
        approximate a field of functions for which the span of the basis is a
        good approximant.

        Returns
        -------
        arby.basis.EIM
            Container for EIM data. Contains (``interpolant``, ``nodes``).

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
        """Compute the squared projection error of a function h onto the basis.

        The error is computed in the L2 norm (continuous case) or the 2-norm
        (discrete case), that is, ||h - P h||^2, where P denotes the projector
        operator associated to the basis.

        Parameters
        ----------
        h : np.ndarray
            Function to be projected.

        Returns
        -------
        error : float
            Square of the projection error.
        """
        diff = h - self.project(h)
        error = self.integration.dot(diff, diff)
        return error

    def project(self, h):
        """Project a function h onto the basis.

        This method represents the action of projecting the function h onto the
        span of the basis.

        Parameters
        ----------
        h : np.ndarray
            Function or set of functions to be projected.

        Returns
        -------
        projected_function : np.ndarray
            Projection of h onto the basis.
        """
        projected_function = 0.0
        for e in self.data:
            projected_function += np.tensordot(
                self.integration.dot(e, h), e, axes=0
            )
        return projected_function

    def interpolate(self, h):
        """Interpolate a function h at EIM nodes.

        This method uses the basis and associated EIM nodes
        (see the ``arby.Basis.eim_`` method) for interpolation.

        Parameters
        ----------
        h : np.ndarray
            Function or set of functions to be interpolated.

        Returns
        -------
        h_interpolated : np.ndarray
            Interpolated function at EIM nodes.
        """
        h = h.transpose()
        h_at_nodes = h[np.array(self.eim_.nodes)]
        h_interpolated = self.eim_.interpolant @ h_at_nodes
        return h_interpolated.transpose()


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


def _sq_proj_errors(training, proj_matrix, basis, dot_product):
    """Square of projection errors from precomputed projection coefficients.

    Parameters
    ----------
    training : numpy.ndarray
    proj_matrix : numpy.ndarray
        Stores the projection coefficients of the training functions.
    basis : numpy.ndarray
        Basis elements.
    dot_product : arby.Integration.dot
        Inherited dot product.

    Returns
    -------
    proj_errors : numpy.ndarray
        Squared projection errors.
    """
    projected_training = proj_matrix.transpose() @ basis
    diff = training - projected_training

    return np.real(dot_product(diff, diff))


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
    algorithm [Hoffmann1989]_ to build an orthonormal basis from a set
    of functions.

    Parameters
    ----------
    functions : array_like, shape=(m, L)
        Functions to be orthonormalized, where `m` is the number of functions
        and `L` is the length of the sample.
    integration : arby.integrals.Integration
        Instance of the `Integration` class for defining inner products.
    max_iter : int, optional
        Maximum number of iterations. Default = 3.

    Returns
    -------
    basis : numpy.ndarray
        Orthonormalized array.

    Raises
    ------
    ValueError
        If functions are not linearly independent for a given tolerance.

    References
    ----------
    .. [Hoffmann1989] Hoffmann, W. Iterative algorithms for Gram-Schmidt
       orthogonalization. Computing 41, 335-348 (1989).
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
) -> RB:
    """Build a reduced basis from training data.

    This function implements the Reduce Basis (RB) greedy algorithm for
    building an orthonormalized reduced basis out from training data. The basis
    is built for reproducing the training functions within a user specified
    tolerance [TiglioAndVillanueva2021]_ by linear combination of its elements.
    Tuning the ``greedy_tol`` parameter allows to control the representation
    accuracy of the basis.

    The ``integration_rule`` parameter specifies the rule that defines inner
    products. If the training functions (rows of the ``training_set``) does not
    correspond to continuous data (e.g. time), choose ``"euclidean"``.
    Otherwise choose any of the quadratures defined in the ``arby.Integration``
    class.

    The output is a container which comprises RB data: a ``basis`` object
    storing the reduced basis and handling tools (see ``arby.Basis``); the
    greedy ``errors`` corresponding to the maxima over the ``training set`` of
    the squared projection errors for each greedy swept; the greedy ``indices``
    locating the most relevant training functions used for building the basis;
    and the ``projection_matrix`` storing the projection coefficients generated
    by the greedy algorithm. For example, we can recover the training set (more
    precisely, a compressed version of it) by multiplying the projection matrix
    with the reduced basis.

    Returns
    -------
    arby.basis.RB
        Container for RB data. Contains (``basis``, ``errors``, ``indices``,
        ``projection_matrix``).

    Raises
    ------
    ValueError
        If ``training_set.shape[1]`` doesn't coincide with quadrature rule
        weights.

    References
    ----------
    .. [TiglioAndVillanueva2021] Reduced Order and Surrogate Models for
       Gravitational Waves. Tiglio, M. and Villanueva A. arXiv:2101.11608
       (2021)

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
    proj_matrix = np.empty((Ntrain, Ntrain), dtype=training_set.dtype)

    # Seed
    greedy_indices = [index_seed]
    basis_data = np.empty_like(training_set)
    basis_data[0] = integration.normalize(training_set[index_seed])

    proj_matrix[0] = integration.dot(basis_data[0], training_set)

    errs = _sq_proj_errors(
        training_set,
        proj_matrix[:1],
        basis_data[:1],
        integration.dot
    )
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
                    data=basis_data[: nn], integration=integration
                ),
                indices=greedy_indices,
                errors=greedy_errors,
                projection_matrix=proj_matrix,
            )

        greedy_indices.append(next_index)
        basis_data[nn], _ = _gs_one_element(
            training_set[greedy_indices[nn]],
            basis_data[:nn],
            integration,
        )
        proj_matrix[nn] = integration.dot(basis_data[nn], training_set)
        errs = _sq_proj_errors(
            training_set,
            proj_matrix[:nn + 1],
            basis_data[:nn + 1],
            integration.dot
        )
        next_index = np.argmax(errs)
        greedy_errors[nn] = errs[next_index]

        sigma = errs[next_index]

        logger.debug(nn, "\t", sigma)

    # Prune excess allocated entries
    greedy_errors, proj_matrix = _prune(greedy_errors, proj_matrix, nn + 1)

    return RB(
        basis=Basis(data=basis_data[: nn + 1], integration=integration),
        indices=greedy_indices,
        errors=greedy_errors,
        projection_matrix=proj_matrix.transpose(),
    )

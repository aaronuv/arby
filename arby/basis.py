# - basis.py -

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE

"""Basis analysis module."""

import functools
import logging

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


@attr.s(frozen=True, hash=False, slots=True)
class EIM:
    """Container for EIM information.

    Parameters
    ----------
    interpolant: numpy.ndarray
        Interpolant matrix.

    nodes: list
        EIM nodes.
    """

    interpolant: np.ndarray = attr.ib()
    nodes: list = attr.ib()


@attr.s(frozen=True, hash=False, slots=True)
class RB:
    """Container for RB information.

    Parameters
    ----------
    basis: np.ndarray
        Reduced basis object.
    indices: np.ndarray
        Greedy indices.
    errors: np.ndarray
        Greedy errors.
    projection_matrix: np.ndarray
        Projection coefficients.

    """

    basis: np.ndarray = attr.ib()
    indices: np.ndarray = attr.ib()
    errors: np.ndarray = attr.ib()
    projection_matrix: np.ndarray = attr.ib()


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
        invV_matrix = np.linalg.inv(v_matrix.T)
        interpolant = self.data.T @ invV_matrix

        return EIM(interpolant=interpolant, nodes=nodes)

    def projection_error(self, h, s=(None,)):
        """Compute the squared projection error of a function h onto the basis.

        The error is computed in the L2 norm (continuous case) or the 2-norm
        (discrete case), that is, ||h - P h||^2, where P denotes the projector
        operator associated to the basis.

        Parameters
        ----------
        h : np.ndarray
            Function to be projected.
        s : tuple, optional
            Slice the basis. If the slice is not provided, the whole basis is
            considered. Default = (None,)

        Returns
        -------
        error : float
            Square of the projection error.
        """
        diff = h - self.project(h, s=s)
        error = self.integration.dot(diff, diff)
        return error

    def project(self, h, s=(None,)):
        """Project a function h onto the basis.

        This method represents the action of projecting the function h onto the
        span of the basis.

        Parameters
        ----------
        h : np.ndarray
            Function or set of functions to be projected.
        s : tuple, optional
            Slice the basis. If the slice is not provided, the whole basis is
            considered. Default = (None,)

        Returns
        -------
        projected_function : np.ndarray
            Projection of h onto the basis.
        """
        s = slice(*s)
        projected_function = 0.0
        for e in self.data[s]:
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
        h = h.T
        h_at_nodes = h[self.eim_.nodes]
        h_interpolated = self.eim_.interpolant @ h_at_nodes
        return h_interpolated.T


# =============================================================================
# INTERNAL FUNCTIONS
# =============================================================================


def _gs_one_element(h, basis, integration, max_iter=3):
    """Orthonormalize a function against an orthonormal basis."""
    norm = integration.norm(h)
    e = h / norm

    for _ in range(max_iter):
        for b in basis:
            e -= b * integration.dot(b, e)
        new_norm = integration.norm(e)
        if new_norm / norm > 0.5:
            break
        norm = new_norm
    else:
        raise StopIteration("Max number of iterations reached ({max_iter}).")

    return e / new_norm, new_norm


def _sq_errs_abs(proj_vector, basis_element, dot_product, diff_training):
    """Square of projection errors from precomputed projection coefficients.

    Since the training set is not a-priori normalized, this function computes
    errors computing the squared norm of the difference between training set
    and the approximation. This method trades accuracy by memory.

    Parameters
    ----------
    proj_vector : numpy.ndarray
        Stores projection coefficients of training functions onto the actual
        basis.
    basis_element : numpy.ndarray
        Actual basis element.
    dot_product : arby.Integration.dot
        Inherited dot product.
    diff_training : numpy.ndarray
        Difference between training set and projected set aiming to be
        actualized.

    Returns
    -------
    proj_errors : numpy.ndarray
        Squared projection errors.
    diff_training : numpy.ndarray
        Actualized difference training set and projected set.
    """
    diff_training = np.subtract(
        diff_training, np.tensordot(proj_vector, basis_element, axes=0)
    )
    return np.real(dot_product(diff_training, diff_training)), diff_training


def _sq_errs_rel(errs, proj_vector):
    """Square of projection errors from precomputed projection coefficients.

    This function takes advantage of an orthonormalized basis and a normalized
    training set to compute fewer floating-point operations than in the
    non-normalized case.

    Parameters
    ----------
    errs : numpy.array
        Projection errors.
    proj_vector : numpy.ndarray
        Stores the projection coefficients of the training set onto the actual
        basis element.

    Returns
    -------
    proj_errors : numpy.ndarray
        Squared projection errors.
    """
    return np.subtract(errs, np.abs(proj_vector) ** 2)


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

    basis_data = np.zeros(functions.shape, dtype=functions.dtype)

    # First element of the basis is special, it's just normalized
    basis_data[0] = integration.normalize(functions[0])

    # For the rest of basis elements add them one by one by extending basis
    def gs_one_element(row):
        return _gs_one_element(
            row, basis=basis_data, integration=integration, max_iter=max_iter
        )[0]

    basis_data[1:] = np.apply_along_axis(
        gs_one_element, axis=1, arr=functions[1:]
    )

    return basis_data


def reduced_basis(
    training_set,
    physical_points,
    integration_rule="riemann",
    greedy_tol=1e-12,
    normalize=False,
) -> RB:
    """Build a reduced basis from training data.

    This function implements the Reduced Basis (RB) greedy algorithm for
    building an orthonormalized reduced basis out from training data. The basis
    is built for reproducing the training functions within a user specified
    tolerance [TiglioAndVillanueva2021]_ by linear combinations of its
    elements. Tuning the ``greedy_tol`` parameter allows to control the
    representation accuracy of the basis.

    The ``integration_rule`` parameter specifies the rule for defining inner
    products. If the training functions (rows of the ``training_set``) does not
    correspond to continuous data (e.g. time), choose ``"euclidean"``.
    Otherwise choose any of the quadratures defined in the ``arby.Integration``
    class.

    Set the boolean ``normalize`` to True if you want to normalize the training
    set before running the greedy algorithm. This condition not only emphasizes
    on structure over scale but may leads to noticeable speedups for large
    datasets.

    The output is a container which comprises RB data: a ``basis`` object
    storing the reduced basis and handling tools (see ``arby.Basis``); the
    greedy ``errors`` corresponding to the maxima over the ``training set`` of
    the squared projection errors for each greedy swept; the greedy ``indices``
    locating the most relevant training functions used for building the basis;
    and the ``projection_matrix`` storing the projection coefficients generated
    by the greedy algorithm. For example, we can recover the training set (more
    precisely, a compressed version of it) by multiplying the projection matrix
    with the reduced basis.

    Parameters
    ----------
    training_set : numpy.ndarray
        The training set of functions.
    physical_points : numpy.ndarray
        Physical points for quadrature rules.
    integration_rule : str, optional
        The quadrature rule to define an integration scheme.
        Default = "riemann".
    greedy_tol : float, optional
        The greedy tolerance as a stopping condition for the reduced basis
        greedy algorithm. Default = 1e-12.
    normalize : bool, optional
        True if you want to normalize the training set. Default = False.

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

    Notes
    -----
    If ``normalize`` is True, the projection coefficients are with respect to
    the original basis but the greedy errors are relative to the normalized
    training set.

    References
    ----------
    .. [TiglioAndVillanueva2021] Reduced Order and Surrogate Models for
       Gravitational Waves. Tiglio, M. and Villanueva A. arXiv:2101.11608
       (2021)

    """
    integration = integrals.Integration(physical_points, rule=integration_rule)

    # useful constants
    Ntrain = training_set.shape[0]
    Nsamples = training_set.shape[1]
    max_rank = min(Ntrain, Nsamples)

    # validate inputs
    if Nsamples != np.size(integration.weights_):
        raise ValueError(
            "Number of samples is inconsistent with quadrature rule."
        )

    if np.allclose(np.abs(training_set), 0, atol=1e-30):
        raise ValueError("Null training set!")

    # ====== Seed the greedy algorithm and allocate memory ======

    # memory allocation
    greedy_errors = np.empty(max_rank, dtype=np.float64)
    proj_matrix = np.empty((max_rank, Ntrain), dtype=training_set.dtype)
    basis_data = np.empty((max_rank, Nsamples), dtype=training_set.dtype)

    norms = integration.norm(training_set)

    if normalize:
        # normalize training set
        training_set = np.array(
            [
                h if np.allclose(h, 0, atol=1e-15) else h / norms[i]
                for i, h in enumerate(training_set)
            ]
        )

        # seed
        next_index = 0
        seed = training_set[next_index]

        while next_index < Ntrain - 1:
            if np.allclose(np.abs(seed), 0):
                next_index += 1
                seed = training_set[next_index]
            else:
                break

        greedy_indices = [next_index]
        basis_data[0] = training_set[next_index]
        proj_matrix[0] = integration.dot(basis_data[0], training_set)
        sq_errors = _sq_errs_rel
        errs = sq_errors(np.ones(Ntrain), proj_matrix[0])

    else:
        next_index = np.argmax(norms)
        greedy_indices = [next_index]
        basis_data[0] = training_set[next_index] / norms[next_index]
        proj_matrix[0] = integration.dot(basis_data[0], training_set)
        sq_errors = _sq_errs_abs
        errs, diff_training = sq_errors(
            proj_matrix[0], basis_data[0], integration.dot, training_set
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
            if normalize:
                # restore proj matrix
                proj_matrix = norms * proj_matrix
            return RB(
                basis=Basis(data=basis_data[:nn], integration=integration),
                indices=greedy_indices,
                errors=greedy_errors,
                projection_matrix=proj_matrix.T,
            )

        greedy_indices.append(next_index)
        basis_data[nn], _ = _gs_one_element(
            training_set[greedy_indices[nn]],
            basis_data[:nn],
            integration,
        )
        proj_matrix[nn] = integration.dot(basis_data[nn], training_set)
        if normalize:
            errs = sq_errors(errs, proj_matrix[nn])
        else:
            errs, diff_training = sq_errors(
                proj_matrix[nn], basis_data[nn], integration.dot, diff_training
            )
        next_index = np.argmax(errs)
        greedy_errors[nn] = errs[next_index]

        sigma = errs[next_index]

        logger.debug(nn, "\t", sigma)

    # Prune excess allocated entries
    greedy_errors, proj_matrix = _prune(greedy_errors, proj_matrix, nn + 1)
    if normalize:
        # restore proj matrix
        proj_matrix = norms * proj_matrix

    return RB(
        basis=Basis(data=basis_data[: nn + 1], integration=integration),
        indices=greedy_indices,
        errors=greedy_errors,
        projection_matrix=proj_matrix.T,
    )

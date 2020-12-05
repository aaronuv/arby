# - core.py -

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE

import numpy as np
import logging
from .integrals import Integration
from random import randint  # noqa: F401
from scipy.interpolate import splrep, splev


# ===================================
#    Iterated-Modified Gram-Schmidt
#      Orthonormalization Function
# ===================================

def gram_schmidt(functions, integration, max_iter=3):
    """Orthonormalize a set of functions.

    This algorithm implements the Iterated, Modified Gram-Schmidt (GS)
    algorithm to build an orthonormal basis from a set of functions
    described in Hoffmann (1989).

    Parameters
    ----------
    functions : array_like, shape=(m, L)
        Functions to be orthonormalized, where m is the number of functions
        and L is the sample length.
    integration : arby.integrals.Integration
        Instance of the Integration class.
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

    Notes
    -----
    Hoffmann, W. Iterative algorithms for Gram-Schmidt orthogonalization.
    Computing 41, 335–348 (1989). https://doi.org/10.1007/BF02241222
    """
    functions = np.asarray(functions)

    _, svds, _ = np.linalg.svd(functions)

    linear_indep_tol = 5e-15
    if min(svds) < linear_indep_tol:
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
    basis = np.array(ortho_basis)

    return basis


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
                raise StopIteration("Gram-Schmidt algorithm: max number of "
                                    "iterations reached ({}).".format(max_iter)
                                    )
        else:
            continue_loop = False

    return e / new_norm, new_norm


# =================================
# Class for Reduced Order Modeling
# =================================

class ReducedOrderModeling:
    """Build reduced order models from training data.

    This class comprises a set of tools to build and manage reduced bases,
    empirical interpolants and predictive models from a pre-computed training
    space of functions. The underlying model g(v,x) describing the training
    space is a real or complex function parameterized by v called the
    "training" parameter. The dual variable x, called the "physical" variable,
    belongs to the domain in which it is defined an inner product.

    Parameters
    ----------
    training_space : array_like, optional
        Array of training functions. Default = None.
    physical_interval : array_like, optional
        Array of physical points. Default = None.
    parameter_interval : array_like, optional
        Array of parameter points. Default = None.
    basis : array_like, optional
        Reduced basis for the Reduced Order Model.
    integration_rule : str, optional
        The quadrature rule to define an integration scheme. Default = Riemann.
    greedy_tol : float, optional
        The greedy tolerance as a stopping condition for the reduced basis
        greedy algorithm. Default = 1e-12.
    poly_deg: int, optional
        Degree of the polynomials used to build splines.
    """
    training_space = None
    """Training functions (numpy.ndarray)."""
    Ntrain = None
    """Number of training functions or parameter points (int)."""
    Nsamples = None
    """Number of sample or physical points (int)."""
    physical_interval = None
    """Sample points (numpy.ndarray)."""
    parameter_interval = None
    """Parameter points (numpy.ndarray)."""
    integration = None
    """Instance of the Integration class (arby.integrals.Integration)."""
    greedy_tol = 1e-12
    """The greedy tolerance (float)."""

    def __init__(self,
                 training_space=None,
                 physical_interval=None,
                 parameter_interval=None,
                 basis=None,
                 integration_rule="riemann",
                 greedy_tol=1e-12,
                 poly_deg=3):
        # Check non empty inputs with the aim of build a reduced order model
        if training_space is not None and physical_interval is not None:
            self.training_space = np.asarray(training_space)
            self.Ntrain, self.Nsamples = self.training_space.shape
            self.physical_interval = np.asarray(physical_interval)
            if self.Ntrain > self.Nsamples:
                raise ValueError("Number of samples must be greater than "
                                 "number of training functions.")
            if self.Nsamples != self.physical_interval.size:
                raise ValueError(
                    "Number of samples for each training function must be "
                    "equal to number of physical points."
                                    )
            if parameter_interval is not None:
                self.parameter_interval = np.asarray(parameter_interval)
                if self.Ntrain != self.parameter_interval.size:
                    raise ValueError(
                        "Number of training functions must be "
                        "equal to number of parameter points."
                    )

            phys_min = self.physical_interval.min()
            phys_max = self.physical_interval.max()
            self.integration = Integration(interval=[phys_min, phys_max],
                                           num=self.Nsamples,
                                           rule=integration_rule
                                           )
        self._greedy_tol = greedy_tol
        self._poly_deg = poly_deg
        self._basis = basis
        self._spline_model = None
        self._logger = logging.getLogger()

    @property
    def greedy_tol(self):
        return self._greedy_tol

    @property
    def poly_deg(self):
        return self._poly_deg

    # ==== Reduced Basis Method ===============================================

    @property
    def basis(self):
        """Array of basis elements.

        If not None, sets a user-specified basis. If None, it implements the
        Reduced Basis greedy algorithm [1] to build an orthonormal basis from
        training data that reproduces the training functions within a tolerance
        specified by the user.

        Returns
        -------
        basis : numpy.ndarray
            The reduced basis of the Reduced Order Model.

        Created Attributes
        ------------------
        Nbasis : int
            Number of basis elements.
        greedy_indices : list(int)
            Set of indices corresponding to greedy points.
        proj_matrix : numpy.ndarray, shape (Nbasis,Ntrain)
            Stores the projection coefficients built in the greedy reduced
            basis Greedy algorithm.

        Raises
        ------
        ValueError
            If Nsamples does not coincide with weights of the quadrature rule.

        Notes
        -----
        [1] Scott E. Field, Chad R. Galley, Jan S. Hesthaven, Jason Kaye, and
        Manuel Tiglio. Fast Prediction and Evaluation of Gravitational
        Waveforms Using Surrogate Models. Phys. Rev. X 4, 031006
        """

        if self._basis is not None:
            self._basis = np.asarray(self._basis)
            self.Nbasis = self._basis.shape[0]
            return self._basis

        self._loss = self._projection_error

        # If seed gives a null function, iterate to a new seed
        index_seed = 0
        seed_function = self.training_space[index_seed]
        zero_function = np.zeros_like(seed_function)
        while np.allclose(seed_function, zero_function):
            index_seed = np.randint(1, self.Ntrain)
            seed_function = self.training_space[index_seed]

        # ====== Seed the greedy algorithm and allocate memory ======

        # Validate inputs
        if self.Nsamples != np.size(self.integration.weights):
            raise ValueError("Number of samples is inconsistent "
                             "with quadrature rule.")

        # Allocate memory for greedy algorithm arrays
        self._allocate(
            self.Ntrain, self.Nsamples, dtype=self.training_space.dtype
        )

        self._norms = self.integration.norm(self.training_space)

        # Seed
        self.greedy_indices = [index_seed]
        self._basis = np.empty_like(self.training_space)
        self._basis[0] = (
            self.training_space[index_seed] / self._norms[index_seed]
        )
        self._basisnorms[0] = self._norms[index_seed]
        self.proj_matrix[0] = self.integration.dot(
            self._basis[0], self.training_space
        )

        errs = self._loss(self.proj_matrix[:1], norms=self._norms)
        next_index = np.argmax(errs)
        self.greedy_errors[0] = errs[next_index]
        sigma = self.greedy_errors[0]

        # ====== Start greedy loop ======
        self._logger.debug("\n Step", "\t", "Error")
        nn = 0
        while sigma > self.greedy_tol:
            nn += 1

            if next_index in self.greedy_indices:
                # Trim excess allocated entries
                self._trim(nn)
                self._basis = self._basis[: nn]
                self.Nbasis = nn
                return self._basis

            self.greedy_indices.append(next_index)
            self._basis[nn], self._basisnorms[nn] = _gs_one_element(
                self.training_space[self.greedy_indices[nn]],
                self._basis[:nn],
                self.integration
            )
            self.proj_matrix[nn] = self.integration.dot(
                self._basis[nn], self.training_space
            )
            errs = self._loss(self.proj_matrix[:nn + 1], norms=self._norms)
            next_index = np.argmax(errs)
            self.greedy_errors[nn] = errs[next_index]

            sigma = errs[next_index]

            self._logger.debug(nn, "\t", sigma)
        # Trim excess allocated entries
        self._trim(nn + 1)
        self._basis = self._basis[:nn + 1]
        self.Nbasis = nn + 1
        return self._basis

    # ====== Empirical Interpolation Method ===================================

    def build_eim(self):
        """Find EIM nodes and build Empirical Interpolant matrix.

        Implements the Empirical Interpolation Method [1] to select a set of
        interpolation nodes from the physical interval and build an interpolant
        matrix.

        Created Attributes
        ------------------
        v_matrix : numpy.ndarray, shape=(Nbasis, Nbasis)
            The Vandermonde matrix associated to the basis.
        interpolant : numpy.ndarray, shape=(Nsamples, Nbasis)
            Interpolant matrix.
        eim_nodes : list(int)
            List of interpolation nodes.

        Raises
        ------
        ValueError
            If there is no basis for EIM.

        Notes
        -----
        [1] Scott E. Field, Chad R. Galley, Jan S. Hesthaven, Jason Kaye, and
        Manuel Tiglio. Fast Prediction and Evaluation of Gravitational
        Waveforms Using Surrogate Models. Phys. Rev. X 4, 031006
        """

        if self.basis is None:
            raise ValueError("There is no basis to work with.")

        nodes = []
        v_matrix = None
        first_node = np.argmax(np.abs(self.basis[0]))
        nodes.append(first_node)

        self._logger.debug(first_node)

        for i in range(1, self.Nbasis):
            v_matrix = self._next_vandermonde(nodes, v_matrix)
            base_at_nodes = [self.basis[i, t] for t in nodes]
            invV_matrix = np.linalg.inv(v_matrix)
            step_basis = self.basis[:i]
            basis_interpolant = base_at_nodes @ invV_matrix @ step_basis
            residual = self.basis[i] - basis_interpolant
            new_node = np.argmax(abs(residual))

            self._logger.debug(new_node)
            nodes.append(new_node)

        v_matrix = np.array(self._next_vandermonde(nodes, v_matrix))
        self.v_matrix = v_matrix.transpose()
        invV_matrix = np.linalg.inv(self.v_matrix)
        self.interpolant = self.basis.transpose() @ invV_matrix
        self.eim_nodes = nodes

    # ==== Surrogate Methods ==================================================

    def surrogate(self, param):
        """Evaluates the surrogate model at a given parameter.

        Invokes the basis property and build_eim methods to build a complete
        surrogate model valid in the entire parameter domain. This is done only
        once, at the first function call. For subsequent calls, the method
        invokes the spline model built at the first call and evaluates. The
        input could be a single parameter point or set of parameters. The
        output is an array storing the surrogate function/s at that/those
        parameter/s with the lenght of the original physical interval sampling.

        Parameters
        ----------
        param : float or array_like(float)
            Point or set of parameters.

        Returns
        -------
        h_surrogate : numpy.ndarray
            The evaluated surrogate function for the given parameters.
        """
        if self._spline_model is None:
            self.build_eim()

            training_compressed = np.empty((self.Ntrain, self.basis.size),
                                           dtype=self.training_space.dtype
                                           )

            for i in range(self.Ntrain):
                for j, node in enumerate(self.eim_nodes):
                    training_compressed[i, j] = self.training_space[i, node]

            h_in_nodes_splined = []
            for i in range(self.Nbasis):
                h_in_nodes_splined.append(
                    splrep(self.parameter_interval,
                           training_compressed[:, i],
                           k=self.poly_deg)
                    )

            self._spline_model = h_in_nodes_splined

        h_surr_at_nodes = np.array(
            [splev(param, spline) for spline in self._spline_model])
        h_surrogate = self.interpolant @ h_surr_at_nodes

        return h_surrogate

    # ==== Private methods ====================================================

    def _projection_error(self, proj_matrix, norms):
        """ Square of projection errors.

        Parameters
        ----------
        proj_matrix : numpy.ndarray, shape=(n,Ntrain)
            Stores the projection coefficients of the training functions. n
            is the number of basis elements of the actual reduced space.
        norms : numpy.ndarray, shape=(Ntrain)
            Stores the norms of the training functions.

        Returns
        -------
        proj_errors : numpy.ndarray, shape=(Ntrain)
            Squared projection errors.
        """
        proj_norms = np.array(
            [np.linalg.norm(proj_matrix[:, i]) for i in range(self.Ntrain)]
        )
        proj_errors = norms ** 2 - proj_norms ** 2
        return proj_errors

    def _allocate(self, Npoints, Nquads, dtype="complex"):
        """Allocate memory for numpy arrays used for building reduced basis."""
        self.greedy_errors = np.empty(Npoints, dtype="double")
        self._basisnorms = np.empty(Npoints, dtype="double")
        self.proj_matrix = np.empty((Npoints, Npoints), dtype=dtype)

    def _trim(self, num):
        """Trim arrays to have size num."""
        self.greedy_errors = self.greedy_errors[:num]
        self.proj_matrix = self.proj_matrix[:num]

    def _next_vandermonde(self, nodes, vandermonde=None):
        """Build the next Vandermonde matrix from the previous one."""
        if vandermonde is None:
            vandermonde = [[self.basis[0, nodes[0]]]]
            return vandermonde

        n = len(vandermonde)
        new_node = nodes[-1]
        for i in range(n):
            vandermonde[i].append(self.basis[i, new_node])
        vertical_vector = [self.basis[n, nodes[j]] for j in range(n)]
        vertical_vector.append(self.basis[n, new_node])
        vandermonde.append(vertical_vector)
        return vandermonde

    # ==== Validation methods =================================================

    def projection_error(self, h, basis):
        """Square of the projection error on a basis of a function or set of
        functions h.
        """
        h_norm = self.integration.norm(h).real
        inner_prod = np.array(
                              [self.integration.dot(basis_elem, h)
                               for basis_elem in basis]
                              )
        return h_norm ** 2 - np.linalg.norm(inner_prod) ** 2

    def project(self, h, basis):
        """Project a function h on a basis."""
        projected_function = 0.0
        for e in basis:
            projected_function += e * self.integration.dot(e, h)
        return projected_function

    def interpolate(self, h):
        """Interpolate a function h at EIM nodes."""
        h_at_nodes = np.array([h[eim_node] for eim_node in self.eim_nodes])
        h_interpolated = self.interpolant @ h_at_nodes
        return h_interpolated

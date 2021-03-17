# - core.py -

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE

"""ROM class and Gram-schmidt function."""

import logging

import attr

import numpy as np

from scipy.interpolate import splev, splrep

from .integrals import Integration, QUADRATURES


# ================
# Constants
# ================


# Set debbuging variable. Don't have actual implementation
logger = logging.getLogger("arby.core")


# ===================================
#    Iterated-Modified Gram-Schmidt
#      Orthonormalization Function
# ===================================


def gram_schmidt(functions, integration, max_iter=3):
    """Orthonormalize a set of functions.

    This algorithm implements the Iterated, Modified Gram-Schmidt (GS)
    algorithm to build an orthonormal basis from a set of functions
    described in [1]_.

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
    .. [1] Hoffmann, W. Iterative algorithms for Gram-Schmidt
      orthogonalization. Computing 41, 335–348 (1989).
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
                raise StopIteration(
                    "Gram-Schmidt algorithm: max number of "
                    "iterations reached ({}).".format(max_iter)
                )
        else:
            continue_loop = False

    return e / new_norm, new_norm


# =================================
# Class for Reduced Order Modeling
# =================================


@attr.s()
class ReducedOrderModel:
    """Build reduced order models from training data.

    This class comprises a set of tools to build and manage reduced bases,
    empirical interpolants and predictive models from a pre-computed training
    space of functions. The underlying model g(v,x) describing the training
    space is a real function parameterized by v called the *training*
    parameter. The dual variable x, called the *physical* variable, belongs
    to the domain in which an inner product is defined. bla

    Parameters
    ----------
    training_space : array_like, optional
        Array of training functions. Default = None.
    physical_interval : array_like, optional
        Array of physical points. Default = None.
    parameter_interval : array_like, optional
        Array of parameter points. Default = None.
    basis : array_like, optional
        Orthonormal basis. It can be specified by the user or built with the
        `basis` property method. Default = None.
    integration_rule : str, optional
        The quadrature rule to define an integration scheme.
        Default = "riemann".
    greedy_tol : float, optional
        The greedy tolerance as a stopping condition for the reduced basis
        greedy algorithm. Default = 1e-12.
    poly_deg: int, optional
        Degree <= 5 of the polynomials used to build splines. Default = 3.

    Attributes
    ----------
    Ntrain_: int
        Number of training functions or parameter points.
    Nsamples_: int
        Number of sample or physical points.
    integration_: arby.integrals.Integration
        Instance of the `Integration` class.
    greedy_indices_: list(int)
        Indices selected by the reduced basis greedy algorithm.
    Nbasis_: int
        Number of basis elements.
    eim_nodes_: list(int)
        Indices selected by the EIM in `build_eim`.
    interpolant_: numpy.ndarray
        Empirical Interpolation matrix.

    Examples
    --------
    **Build a surrogate model**

    >>> from arby import ReducedOrderModel as ROM

    Input the three most important parameters.

    >>> model = ROM(training_space, physical_interval, parameter_interval)

    Build and evaluate the surrogate model. The building stage is done once and
    for all. It could take some time. For this reason this stage is called the
    *offline* stage. Subsequent calls will invoke the built surrogate
    spline model and then evaluate. This is called the *online* stage.

    >>> model.surrogate(parameter)

    To improve the accuracy of the model without the addition of more training
    functions, you can tune the class parameters `greedy_tol` and `poly_deg` to
    control the precision of the reduced basis or the spline model.
    """

    training_space: np.ndarray = attr.ib(
        default=None, converter=attr.converters.optional(np.asarray)
    )
    physical_interval: np.ndarray = attr.ib(
        default=None, converter=attr.converters.optional(np.asarray)
    )
    parameter_interval: np.ndarray = attr.ib(
        default=None, converter=attr.converters.optional(np.asarray)
    )

    integration_rule: str = attr.ib(
        default="riemann", validator=attr.validators.in_(QUADRATURES)
    )
    greedy_tol: float = attr.ib(default=1e-12)
    poly_deg: int = attr.ib(default=3)

    _basis: np.ndarray = attr.ib(default=None)

    _spline_model = attr.ib(default=None, init=False)

    Ntrain_: int = attr.ib(init=False)
    Nsamples_: int = attr.ib(init=False)
    integration_: Integration = attr.ib(init=False)

    @Ntrain_.default
    def _Ntrain__default(self):
        if self.training_space is not None:
            return self.training_space.shape[0]

    @Nsamples_.default
    def _Nsamples__default(self):
        if self.training_space is not None:
            return self.training_space.shape[1]

    @integration_.default
    def _integration__default(self):
        if (
            self.training_space is not None
            and self.physical_interval is not None  # noqa: W503
        ):
            return Integration(
                interval=self.physical_interval, rule=self.integration_rule
            )

    def __attrs_post_init__(self):  # noqa all the complex validators
        if (
            self.training_space is not None
            and self.physical_interval is not None  # noqa: W503
        ):
            if self.Ntrain_ > self.Nsamples_:
                raise ValueError(
                    "Number of samples must be greater than "
                    "number of training functions."
                )
            if self.Nsamples_ != self.physical_interval.size:
                raise ValueError(
                    "Number of samples for each training function must be "
                    "equal to number of physical points."
                )
            if self.parameter_interval is not None:
                if self.Ntrain_ != self.parameter_interval.size:
                    raise ValueError(
                        "Number of training functions must be "
                        "equal to number of parameter points."
                    )

    # ==== Reduced Basis Method ===============================================

    def _prune(self, greedy_errors, proj_matrix, num):
        """Prune arrays to have size num."""
        return greedy_errors[:num], proj_matrix[:num]

    @property
    def basis(self):
        """Array of basis elements.

        Return a user-specified basis or implement the Reduced Basis greedy
        algorithm [2]_ to build an orthonormal basis from training data. This
        basis reproduces the training functions by means of projection within a
        tolerance specified by the user.

        Returns
        -------
        basis : numpy.ndarray
            The reduced basis of the Reduced Order Model.

        Raises
        ------
        ValueError
            If ``Nsamples`` doesn't coincide with weights of the quadrature
            rule.

        References
        ----------
        .. [2] Scott E. Field, Chad R. Galley, Jan S. Hesthaven, Jason Kaye,
            and Manuel Tiglio. Fast Prediction and Evaluation of Gravitational
            Waveforms Using Surrogate Models. Phys. Rev. X 4, 031006

        """
        if self._basis is not None:
            self._basis = np.asarray(self._basis)
            self.Nbasis_ = self._basis.shape[0]
            return self._basis

        # If seed gives a null function, choose a random seed
        index_seed = 0
        seed_function = self.training_space[index_seed]
        zero_function = np.zeros_like(seed_function)
        while np.allclose(seed_function, zero_function):
            index_seed = np.random.randint(1, self.Ntrain_)
            seed_function = self.training_space[index_seed]

        # ====== Seed the greedy algorithm and allocate memory ======

        # Validate inputs
        if self.Nsamples_ != np.size(self.integration_.weights_):
            raise ValueError(
                "Number of samples is inconsistent " "with quadrature rule."
            )

        # Allocate memory for greedy algorithm arrays
        greedy_errors = np.empty(self.Ntrain_, dtype="double")
        basisnorms = np.empty(self.Ntrain_, dtype="double")
        self._proj_matrix = np.empty(
            (self.Ntrain_, self.Ntrain_), dtype=self.training_space.dtype
        )

        norms = self.integration_.norm(self.training_space)

        # Seed
        self.greedy_indices_ = [index_seed]
        self._basis = np.empty_like(self.training_space)
        self._basis[0] = self.training_space[index_seed] / norms[index_seed]
        basisnorms[0] = norms[index_seed]
        self._proj_matrix[0] = self.integration_.dot(
            self._basis[0], self.training_space
        )

        errs = self._loss(self._proj_matrix[:1], norms=norms)
        next_index = np.argmax(errs)
        greedy_errors[0] = errs[next_index]
        sigma = greedy_errors[0]

        # ====== Start greedy loop ======
        logger.debug("\n Step", "\t", "Error")
        nn = 0
        while sigma > self.greedy_tol:
            nn += 1

            if next_index in self.greedy_indices_:
                # Prune excess allocated entries
                greedy_errors, self._proj_matrix = self._prune(
                    greedy_errors, self._proj_matrix, nn
                )
                self._basis = self._basis[:nn]
                self.Nbasis_ = nn
                return self._basis

            self.greedy_indices_.append(next_index)
            self._basis[nn], basisnorms[nn] = _gs_one_element(
                self.training_space[self.greedy_indices_[nn]],
                self._basis[:nn],
                self.integration_,
            )
            self._proj_matrix[nn] = self.integration_.dot(
                self._basis[nn], self.training_space
            )
            errs = self._loss(self._proj_matrix[: nn + 1], norms=norms)
            next_index = np.argmax(errs)
            greedy_errors[nn] = errs[next_index]

            sigma = errs[next_index]

            logger.debug(nn, "\t", sigma)
        # Prune excess allocated entries
        greedy_errors, self._proj_matrix = self._prune(
            greedy_errors, self._proj_matrix, nn + 1
        )

        self.greedy_errors = greedy_errors
        self._basis = self._basis[: nn + 1]
        self.Nbasis_ = nn + 1
        return self._basis

    # ====== Empirical Interpolation Method ===================================

    def build_eim(self):
        """Find the EIM nodes and build an Empirical Interpolantion matrix.

        Implement the Empirical Interpolation Method [3]_ to select a set of
        interpolation nodes from the physical interval and build an interpolant
        matrix.

        Raises
        ------
        ValueError
            If there is no basis for EIM.

        References
        ----------
        .. [3] Scott E. Field, Chad R. Galley, Jan S. Hesthaven, Jason Kaye,
          and Manuel Tiglio. Fast Prediction and Evaluation of Gravitational
          Waveforms Using Surrogate Models. Phys. Rev. X 4, 031006

        """
        nodes = []
        v_matrix = None
        first_node = np.argmax(np.abs(self.basis[0]))
        nodes.append(first_node)

        logger.debug(first_node)

        for i in range(1, self.Nbasis_):
            v_matrix = self._next_vandermonde(nodes, v_matrix)
            base_at_nodes = [self.basis[i, t] for t in nodes]
            invV_matrix = np.linalg.inv(v_matrix)
            step_basis = self.basis[:i]
            basis_interpolant = base_at_nodes @ invV_matrix @ step_basis
            residual = self.basis[i] - basis_interpolant
            new_node = np.argmax(abs(residual))

            logger.debug(new_node)
            nodes.append(new_node)

        v_matrix = np.array(self._next_vandermonde(nodes, v_matrix))
        invV_matrix = np.linalg.inv(v_matrix.transpose())
        self.interpolant_ = self.basis.transpose() @ invV_matrix
        self.eim_nodes_ = nodes

    # ==== Surrogate Method ===================================================

    def surrogate(self, param):
        """Evaluate the surrogate model at a given parameter.

        Build a complete surrogate model valid in the entire parameter domain.
        This is done only once, at the first function call. For subsequent
        calls, the method invokes the spline model built at the first call and
        evaluates. The output is an array storing the surrogate function/s at
        that/those parameter/s with the lenght of the original physical
        interval sampling.

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

            training_compressed = np.empty(
                (self.Ntrain_, self.basis.size),
                dtype=self.training_space.dtype,
            )

            for i in range(self.Ntrain_):
                for j, node in enumerate(self.eim_nodes_):
                    training_compressed[i, j] = self.training_space[i, node]

            h_in_nodes_splined = []
            for i in range(self.Nbasis_):
                h_in_nodes_splined.append(
                    splrep(
                        self.parameter_interval,
                        training_compressed[:, i],
                        k=self.poly_deg,
                    )
                )

            self._spline_model = h_in_nodes_splined

        h_surr_at_nodes = np.array(
            [splev(param, spline) for spline in self._spline_model]
        )
        h_surrogate = self.interpolant_ @ h_surr_at_nodes

        return h_surrogate

    # ==== Private methods ====================================================

    def _loss(self, proj_matrix, norms):
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
            [np.linalg.norm(proj_matrix[:, i]) for i in range(self.Ntrain_)]
        )
        proj_errors = norms ** 2 - proj_norms ** 2
        return proj_errors

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
        h_norm = self.integration_.norm(h).real
        inner_prod = np.array(
            [self.integration_.dot(basis_elem, h) for basis_elem in basis]
        )
        l2_error = h_norm ** 2 - np.linalg.norm(inner_prod) ** 2
        return l2_error

    def project(self, h, basis):
        """Project a function h on a basis.

        Parameters
        ----------
        h: numpy.array
            Function or set of functions to be projected.
        basis: numpy.array
            Orthonormal basis.

        Returns
        -------
        projected_function: numpy.array
            Projection of h on the given basis.
        """
        projected_function = 0.0
        for e in basis:
            projected_function += e * self.integration_.dot(e, h)
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
        h_at_nodes = np.array([h[eim_node] for eim_node in self.eim_nodes_])
        h_interpolated = self.interpolant_ @ h_at_nodes
        return h_interpolated

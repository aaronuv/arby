# - rom.py -

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE

"""ROM class and Gram-schmidt function."""

import functools
import logging

import attr

import numpy as np

from scipy.interpolate import splev, splrep

from . import basis, integrals


# ================
# Constants
# ================


# Set debbuging variable. Don't have actual implementation
logger = logging.getLogger("arby.rom")


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
        projected_element, _ = basis._gs_one_element(  # noqa
            new_basis_elem, ortho_basis, integration, max_iter
        )
        ortho_basis.append(projected_element)
    basis = np.array(ortho_basis)

    return basis


# =================================
# Class for Reduced Order Modeling
# =================================


@attr.s(frozen=True, hash=True)
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

    training_space: np.ndarray = attr.ib(converter=np.asarray)
    physical_interval: np.ndarray = attr.ib(converter=np.asarray)
    parameter_interval: np.ndarray = attr.ib(converter=np.asarray)

    integration_rule: str = attr.ib(
        default="riemann", validator=attr.validators.in_(integrals.QUADRATURES)
    )
    greedy_tol: float = attr.ib(default=1e-12)
    poly_deg: int = attr.ib(default=3)

    # ==== Size properties ==============================================

    @property
    def Ntrain_(self):
        return self.training_space.shape[0]

    @property
    def Nsamples_(self):
        return self.training_space.shape[1]

    # ==== Attrs orchestration ===========================================

    def __attrs_post_init__(self):  # noqa all the complex validators
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

    # ==== Reduced Basis  ===============================================

    @functools.lru_cache(maxsize=None)
    def _basis_and_error(self):
        reduced_basis, greedy_error = basis.reduce_basis(
            self.training_space,
            self.physical_interval,
            self.integration_rule,
            self.greedy_tol,
        )
        return reduced_basis, greedy_error

    @property
    def basis_(self):
        reduced_basis, _ = self._basis_and_error()
        return reduced_basis

    @property
    def greedy_error_(self):
        _, greedy_error = self._basis_and_error()
        return greedy_error

    # ==== Surrogate Method =============================================

    @functools.lru_cache(maxsize=None)
    def _spline_model(self):

        training_compressed = np.empty(
            (self.Ntrain_, self.basis.size_),
            dtype=self.training_space.dtype,
        )

        basis = self.basis_

        for i in range(self.Ntrain_):
            for j, node in enumerate(basis.eim_.nodes):
                training_compressed[i, j] = self.training_space[i, node]

        h_in_nodes_splined = []
        for i in range(self.basis.Nbasis_):
            h_in_nodes_splined.append(
                splrep(
                    self.parameter_interval,
                    training_compressed[:, i],
                    k=self.poly_deg,
                )
            )

        return h_in_nodes_splined

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

        spline_model = self._spline_model()
        basis = self.basis_

        h_surr_at_nodes = np.array(
            [splev(param, spline) for spline in spline_model]
        )
        h_surrogate = basis.eim_.interpolant @ h_surr_at_nodes

        return h_surrogate

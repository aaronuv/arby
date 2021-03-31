# - rom.py -

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE

"""ROM class and Gram-schmidt function."""

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
        """Return the number of training functions or parameter points."""
        return self.training_space.shape[0]

    @property
    def Nsamples_(self):
        """Return the number of sample or physical points."""
        return self.training_space.shape[1]

    # ==== Attrs orchestration ===========================================

    def __attrs_post_init__(self):  # noqa all the complex validators
        if self.Ntrain_ > self.Nsamples_:
            raise ValueError(
                "Number of samples must be greater than "
                "number of training functions. "
                f"{self.Nsamples_} <= {self.Ntrain_}"
            )
        if self.Nsamples_ != self.physical_interval.size:
            raise ValueError(
                "Number of samples for each training function must be "
                "equal to number of physical points. "
                f"{self.Nsamples_} != {self.physical_interval.size}"
            )
        if self.parameter_interval is not None:
            if self.Ntrain_ != self.parameter_interval.size:
                raise ValueError(
                    "Number of training functions must be "
                    "equal to number of parameter points. "
                    f"{self.Ntrain_} != {self.parameter_interval.size}"
                )

    # ==== Reduced Basis  ===============================================

    def _basis_and_error(self):
        if not hasattr(self, "_cached_basis_and_error"):
            reduced_basis, greedy_error = basis.reduce_basis(
                self.training_space,
                self.physical_interval,
                self.integration_rule,
                self.greedy_tol,
            )
            super().__setattr__(
                "_cached_basis_and_error", (reduced_basis, greedy_error)
            )

        return self._cached_basis_and_error

    @property
    def basis_(self):
        """Reduced Basis greedy algorithm implementation."""
        reduced_basis, _ = self._basis_and_error()
        return reduced_basis

    @property
    def greedy_error_(self):
        """Error of the reduce basis greedy algorithm."""
        _, greedy_error = self._basis_and_error()
        return greedy_error

    # ==== Surrogate Method =============================================

    def _spline_model(self):
        if not hasattr(self, "_cached_spline_model"):
            basis = self.basis_

            training_compressed = np.empty(
                (self.Ntrain_, basis.size_),
                dtype=self.training_space.dtype,
            )

            for i in range(self.Ntrain_):
                for j, node in enumerate(basis.eim_.nodes):
                    training_compressed[i, j] = self.training_space[i, node]

            h_in_nodes_splined = []
            for i in range(basis.Nbasis_):
                h_in_nodes_splined.append(
                    splrep(
                        self.parameter_interval,
                        training_compressed[:, i],
                        k=self.poly_deg,
                    )
                )
            super().__setattr__("_cached_spline_model", h_in_nodes_splined)

        return self._cached_spline_model

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

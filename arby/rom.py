# - rom.py -

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE

"""Reduced Order Modeling module."""

import logging

import attr

import numpy as np

from scipy.interpolate import splev, splrep

from . import basis, integrals


# ================
# Constants
# ================


# Set debugging variable. Don't have actual implementation
logger = logging.getLogger("arby.rom")


# =================================
# Class for Reduced Order Modeling
# =================================


@attr.s(frozen=True, hash=True)
class ReducedOrderModel:
    """Build reduced order models from training data.

    This class comprises a set of tools to build and handle reduced bases,
    empirical interpolants and predictive models from pre-computed training
    set of functions. The underlying or ground truth model describing the
    training set is a real function g(v,x) parameterized by a *training*
    parameter v. The *physical* variable x belongs to a domain for which an
    inner product can defined. The surrogate model is built bringing together
    the Reduced Basis (RB) greedy algorithm and the Empirical Interpolation
    Method (EIM) to work in synergy towards a predictive model for the ground
    truth model.

    Parameters
    ----------
    training_set : array_like
        Array of training functions.
    physical_points : array_like
        Array of physical points.
    parameter_points : array_like
        Array of parameter points.
    integration_rule : str, optional
        The quadrature rule to define an integration scheme.
        Default = "riemann".
    greedy_tol : float, optional
        The greedy tolerance as a stopping condition for the reduced basis
        greedy algorithm. Default = 1e-12.
    poly_deg: int, optional
        Degree <= 5 of polynomials used for splines. Default = 3.

    --> Attributes

    Nsamples_: int
        Number of physical points.
    Ntrain_: int
        Number of training functions or parameter points.
    basis_: arby.basis.Basis
        Basis object comprising the reduced basis and handling tools.
    greedy_indices_: tuple
        Greedy indices from the RB algorithm.
    greedy_errors_: np.ndarray
        Greedy projection errors from the RB algorithm.
    projection_matrix_: np.ndarray
        Matrix of projection coefficients from the RB algorithm.
    eim_: arby.basis.EIM
        Container for EIM information. Stores the `interpolant` matrix and
        the EIM `nodes` given by the EIM algorithm.
    -->

    Examples
    --------
    **Build a surrogate model**

    >>> from arby import ReducedOrderModel as ROM

    Input the three most important parameters (the others are optional).

    >>> model = ROM(training_set, physical_points, parameter_points)

    Build/evaluate the surrogate model. The building stage is done once and
    for all at the first call. It could take some time for large training
    sets. For this reason it is called the *offline* stage. Subsequent calls
    will invoke the already built surrogate model and just evaluates it. That
    corresponds to the *online* stage.

    >>> model.surrogate(parameter)

    For attempting to improve the model's accuracy without the addition of more
    training functions, tune the class parameters ``greedy_tol`` and
    ``poly_deg`` to control the precision of the reduced basis
    (see the ``arby.reduced_basis`` method) or the internal splines model.

    """

    training_set: np.ndarray = attr.ib(converter=np.asarray)
    physical_points: np.ndarray = attr.ib(converter=np.asarray)
    parameter_points: np.ndarray = attr.ib(converter=np.asarray)

    integration_rule: str = attr.ib(
        default="riemann", validator=attr.validators.in_(integrals.QUADRATURES)
    )
    greedy_tol: float = attr.ib(default=1e-12)
    poly_deg: int = attr.ib(default=3)

    # ==== Size properties ==============================================

    @property
    def Ntrain_(self):
        """Return the number of training functions or parameter points."""
        return self.training_set.shape[0]

    @property
    def Nsamples_(self):
        """Return the number of samples or physical points."""
        return self.training_set.shape[1]

    # ==== Attrs orchestration ===========================================

    def __attrs_post_init__(self):  # noqa all the complex validators
        if self.Nsamples_ != self.physical_points.size:
            raise ValueError(
                "Number of samples for each training function must be "
                "equal to number of physical points. "
                f"{self.Nsamples_} != {self.physical_points.size}"
            )
        if self.parameter_points is not None:
            if self.Ntrain_ != self.parameter_points.size:
                raise ValueError(
                    "Number of training functions must be "
                    "equal to number of parameter points. "
                    f"{self.Ntrain_} != {self.parameter_points.size}"
                )

    # ==== Reduced Basis and EIM =========================================

    def _rbalg_outputs(self):
        if not hasattr(self, "_cached_rbalg_outputs"):
            rbalg_outputs = basis.reduced_basis(
                self.training_set,
                self.physical_points,
                self.integration_rule,
                self.greedy_tol,
            )
            super().__setattr__("_cached_rbalg_outputs", rbalg_outputs)

        return self._cached_rbalg_outputs

    @property
    def basis_(self):
        """Return a reduced basis object.

        The reduced basis is computed at the first call and stored as a class
        object of ``arby.Basis``, which comprises several tools for handling
        bases. See also the ``arby.reduced_basis`` documentation.

        """
        return self._rbalg_outputs().basis

    @property
    def greedy_indices_(self):
        """Greedy indices from the RB algorithm.

        See the ``arby.reduced_basis`` documentation.

        """
        return self._rbalg_outputs().indices

    @property
    def greedy_errors_(self):
        """Errors computed in the RB algorithm.

        See the ``arby.reduced_basis`` documentation.

        """
        return self._rbalg_outputs().errors

    @property
    def projection_matrix_(self):
        """Projection coefficients from the RB algorithm.

        See the ``arby.reduced_basis`` documentation.

        """
        return self._rbalg_outputs().projection_matrix

    @property
    def eim_(self):
        """Return EIM data.

        See ``arby.Basis.eim_`` documentation.

        """
        return self.basis_.eim_

    # ==== Surrogate Method =============================================

    def _spline_model(self):
        if not hasattr(self, "_cached_spline_model"):
            basis = self.basis_
            eim = self.eim_

            training_compressed = np.empty(
                (self.Ntrain_, basis.Nbasis_),
                dtype=self.training_set.dtype,
            )

            for i in range(self.Ntrain_):
                for j, node in enumerate(eim.nodes):
                    training_compressed[i, j] = self.training_set[i, node]

            h_in_nodes_splined = [
                splrep(
                    self.parameter_points,
                    training_compressed[:, i],
                    k=self.poly_deg,
                )
                for i, _ in enumerate(basis.data)
            ]

            super().__setattr__("_cached_spline_model", h_in_nodes_splined)

        return self._cached_spline_model

    def surrogate(self, param):
        """Evaluate the surrogate model at parameter/s.

        Build a surrogate model valid for the entire parameter domain.
        The building stage is performed only once for the first function call.
        For subsequent calls, the method invokes the already fitted model and
        just evaluates it. The output is an array storing surrogate evaluations
        at the parameter/s.

        Parameters
        ----------
        param : float or array_like(float)
            Point or set of parameters inside the parameter domain.

        Returns
        -------
        h_surrogate : numpy.ndarray
            The evaluated surrogate function for the given parameters.

        """
        spline_model = self._spline_model()
        eim = self.eim_

        h_surr_at_nodes = np.array(
            [splev(param, spline) for spline in spline_model]
        )
        h_surrogate = eim.interpolant @ h_surr_at_nodes

        return h_surrogate

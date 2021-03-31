# test_arby.py

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================
import arby

import numpy as np

from scipy.special import jv as BesselJ


# =============================================================================
# TESTS
# =============================================================================


def test_basis_shape(basis_data, physical_interval):
    integration = arby.Integration(physical_interval)
    basis = arby.Basis(basis_data, integration)

    assert basis.Nbasis_ == len(basis_data)


def test_eim(basis_data, physical_interval):
    integration = arby.Integration(physical_interval)
    basis = arby.Basis(basis_data, integration)

    np.testing.assert_allclose(basis.eim_.interpolant.mean(), 0.1)
    np.testing.assert_allclose(basis.eim_.interpolant.std(), 0.345161095)
    np.testing.assert_array_equal(
        basis.eim_.nodes, [0, 100, 2, 36, 9, 72, 1, 20, 89, 4]
    )


def test_projection_error(basis_data, physical_interval):
    integration = arby.Integration(physical_interval)
    basis = arby.Basis(basis_data, integration)

    perror = basis.projection_error(physical_interval)
    np.testing.assert_almost_equal(perror, 0, decimal=10)


def test_project(basis_data, physical_interval):
    integration = arby.Integration(physical_interval)
    basis = arby.Basis(basis_data, integration)

    project = basis.project(physical_interval)
    np.testing.assert_allclose(
        project, physical_interval, rtol=1e-4, atol=1e-8
    )


def test_interpolate(basis_data, physical_interval):
    integration = arby.Integration(physical_interval)
    basis = arby.Basis(basis_data, integration)

    interpolation = basis.interpolate(physical_interval)
    np.testing.assert_allclose(
        interpolation, physical_interval, rtol=1e-4, atol=1e-8
    )


def test_reduce_basis(training_space):
    physical_interval = np.linspace(0, 1, 101)

    basis, error = arby.reduce_basis(training_space, physical_interval)

    assert len(basis.data) == 9
    np.testing.assert_allclose(basis.data.mean(), 0.136109, atol=1e-6)
    np.testing.assert_allclose(basis.data.std(), 1.005630, atol=1e-6)

    assert basis.eim_.nodes == (0, 100, 2, 36, 9, 72, 1, 20, 89)

    assert len(basis.eim_.interpolant) == 101
    np.testing.assert_allclose(
        basis.eim_.interpolant.mean(), 0.111111, atol=1e-6
    )
    np.testing.assert_allclose(
        basis.eim_.interpolant.std(), 0.351107, atol=1e-6
    )

    np.testing.assert_allclose(error.mean(), 0.0047, atol=1e-6)
    np.testing.assert_allclose(error.std(), 0.011576, atol=1e-6)


def test_projectors():
    """Test that projectors works as true projectors."""
    random = np.random.default_rng(seed=42)

    nu_train = np.linspace(1, 10, num=101)
    physical_interval = np.linspace(0, 1, 1001)

    training = np.array([BesselJ(nn, physical_interval) for nn in nu_train])

    basis, _ = arby.reduce_basis(training, physical_interval, greedy_tol=1e-12)

    # compute a random index to test Proj_operator^2 = Proj_operator
    sample = random.choice(training)
    proj_fun = basis.project(sample)
    re_proj_fun = basis.project(proj_fun)

    np.testing.assert_allclose(proj_fun, re_proj_fun, rtol=1e-5, atol=1e-8)


def test_interpolators():
    """Test that projectors works as true projectors."""
    random = np.random.default_rng(seed=42)

    nu_train = np.linspace(1, 10, num=101)
    physical_interval = np.linspace(0, 1, 1001)

    training = np.array([BesselJ(nn, physical_interval) for nn in nu_train])

    basis, _ = arby.reduce_basis(training, physical_interval, greedy_tol=1e-12)

    # compute a random index to test Proj_operator^2 = Proj_operator
    sample = random.choice(training)

    interp_fun = basis.interpolate(sample)
    re_interp_fun = basis.interpolate(interp_fun)
    np.testing.assert_allclose(interp_fun, re_interp_fun, rtol=1e-5, atol=1e-8)


def test_projection_error_consistency():
    """Test auto-consistency for projection error function."""

    nu_train = np.linspace(1, 10, num=101)
    physical_interval = np.linspace(0, 1, 1001)

    # build traning space
    training = np.array([BesselJ(nn, physical_interval) for nn in nu_train])

    # build reduced basis
    basis, _ = arby.reduce_basis(training, physical_interval, greedy_tol=1e-12)

    # Check that projection errors of basis elements onto the basis is
    # zero
    computed_errors = [
        basis.projection_error(basis_element) for basis_element in basis.data
    ]
    expected_errors = [0.0] * basis.Nbasis_
    np.testing.assert_allclose(
        computed_errors, expected_errors, rtol=1e-5, atol=1e-8
    )

import pytest
@pytest.mark.xfail
def test_greedy_already_selected():
    """Test greedy stopping condition."""
    npoints = 101

    # Sample parameter nu and physical variable x
    nu = np.linspace(0, 10, num=npoints)
    x = np.linspace(0, 1, 101)

    # build traning space
    training = np.array([BesselJ(nn, x) for nn in nu])

    # build reduced basis with exagerated greedy_tol
    bessel = arby.ReducedOrderModel(training, x, greedy_tol=1e-20)

    np.testing.assert_allclose(bessel.basis.mean(), 0.10040373410299859)
    np.testing.assert_allclose(bessel.basis.std(), 1.017765970259045)
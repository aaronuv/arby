# test_arby.py

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================
import arby

import numpy as np

import pytest

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


def test_port(training):
    x = np.linspace(0, 1, 101)
    bessel = arby.ReducedOrderModel(training, x)
    bessel.build_eim()

    original_basis = bessel.basis
    original_error = bessel.greedy_errors_
    original_eim_nodes = bessel.eim_nodes_
    original_interpolant = bessel.interpolant_

    basis, error = arby.reduce_basis(
        bessel.training_space, bessel.physical_interval)

    assert np.all(basis.data == original_basis)
    assert np.all(error == original_error)
    assert np.all(basis.eim_.nodes == original_eim_nodes)
    assert np.all(basis.eim_.interpolant == original_interpolant)


@pytest.mark.xfail
def test_projectors():
    """Test that projectors works as true projectors."""
    npoints = 101
    nu_train = np.linspace(1, 10, num=npoints)
    x = np.linspace(0, 1, 1001)

    # build traning space
    training = np.array([BesselJ(nn, x) for nn in nu_train])

    # build reduced basis
    bessel = arby.ReducedOrderModel(
        training_space=training,
        physical_interval=x,
        parameter_interval=nu_train,
        greedy_tol=1e-12,
    )

    # compute a random index to test Proj_operator^2 = Proj_operator
    random_index = np.random.randint(0, npoints)
    proj_fun = bessel.project(training[random_index], bessel.basis)
    proj2_fun = bessel.project(proj_fun, bessel.basis)
    np.testing.assert_allclose(proj_fun, proj2_fun, rtol=1e-5, atol=1e-8)

    # compute a random index to test Interpolant^2 = Interpolant
    random_index = np.random.randint(0, npoints)

    # build interpolant matrix
    bessel.build_eim()
    interp_fun = bessel.interpolate(training[random_index])
    interp2_fun = bessel.interpolate(interp_fun)
    np.testing.assert_allclose(interp_fun, interp2_fun, rtol=1e-5, atol=1e-8)


@pytest.mark.xfail
def test_projection_error_consistency():
    """Test auto-consistency for projection error function."""
    npoints = 101
    nu_train = np.linspace(1, 10, num=npoints)
    x = np.linspace(0, 1, 1001)

    # build traning space
    training = np.array([BesselJ(nn, x) for nn in nu_train])

    # build reduced basis
    bessel = arby.ReducedOrderModel(
        training_space=training,
        physical_interval=x,
        parameter_interval=nu_train,
        greedy_tol=1e-12,
    )

    # Check that projection errors of basis elements onto the basis is
    # zero
    computed_errors = [
        bessel.projection_error(basis_element, bessel.basis)
        for _, basis_element in enumerate(bessel.basis)
    ]
    expected_errors = [0.0] * bessel.Nbasis_
    np.testing.assert_allclose(
        computed_errors, expected_errors, rtol=1e-5, atol=1e-8
    )

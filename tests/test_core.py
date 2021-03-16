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


def test_sliced_training(training):
    Nsamples = 10
    x = np.linspace(0, 1, 101)
    sliced_training = training[:, :Nsamples]
    with pytest.raises(ValueError):
        arby.ReducedOrderModel(sliced_training, x)


def test_wrong_Nsamples(training):
    wrong_Nsamples = 11
    x = np.linspace(0, 1, wrong_Nsamples)
    with pytest.raises(ValueError):
        arby.ReducedOrderModel(training, x)


def test_wrong_Ntrain(training):
    wrong_Ntrain = 11
    x = np.linspace(0, 1, 101)
    nu = np.linspace(0, 10, wrong_Ntrain)
    with pytest.raises(ValueError):
        arby.ReducedOrderModel(training, x, nu)


def test_alter_Nsamples(training):
    x = np.linspace(0, 1, 101)
    bessel = arby.ReducedOrderModel(training, x)
    bessel.Nsamples_ += 1
    with pytest.raises(ValueError):
        bessel.basis


def test_basis_shape():
    """Test correct shape for reduced basis."""
    npoints = 101
    # Sample parameter nu and physical variable x
    nu = np.linspace(0, 10, num=npoints)
    x = np.linspace(0, 1, 101)
    # build traning space
    training = np.array([BesselJ(nn, x) for nn in nu])
    # build reduced basis
    bessel = arby.ReducedOrderModel(training, x, greedy_tol=1e-12)

    # Assert that basis has correct shape
    assert bessel.basis.ndim == 2
    assert bessel.basis.shape[1] == npoints


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


def test_surrogate_accuracy():
    """Test surrogate accuracy."""
    npoints = 101
    nu_train = np.linspace(1, 10, num=npoints)
    nu_validation = np.linspace(1, 10, num=1001)
    x = np.linspace(0, 1, 1001)
    # build traning space
    training = np.array([BesselJ(nn, x) for nn in nu_train])
    # build reduced basis
    bessel = arby.ReducedOrderModel(
        training_space=training,
        physical_interval=x,
        parameter_interval=nu_train,
        greedy_tol=1e-15,
    )
    bessel_test = [BesselJ(nn, x) for nn in nu_validation]
    bessel_surrogate = [bessel.surrogate(nn) for nn in nu_validation]

    np.testing.assert_allclose(
        bessel_test, bessel_surrogate, rtol=1e-5, atol=1e-5
    )


def test_gram_schmidt():
    """Test Gram Schmidt orthonormalization algorithm."""
    expected_basis = np.loadtxt("tests/bessel/bessel_basis.txt")
    nbasis, npoints = expected_basis.shape
    x = np.linspace(0, 1, 101)
    integration = arby.Integration(interval=x, rule="riemann")
    computed_basis = arby.gram_schmidt(expected_basis, integration)

    np.testing.assert_allclose(
        computed_basis, expected_basis, rtol=1e-5, atol=1e-8
    )


def test_gram_schmidt_linear_independence():
    expected_basis = np.loadtxt("tests/bessel/bessel_basis.txt")
    nbasis, npoints = expected_basis.shape
    x = np.linspace(0, 1, 101)
    integration = arby.Integration(interval=x, rule="riemann")

    non_li_functions = expected_basis
    non_li_functions[0] = expected_basis[1]

    # non linear independence error capturing
    with pytest.raises(ValueError):
        arby.gram_schmidt(expected_basis, integration)


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


def test_projection_error():
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

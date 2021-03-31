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


def test_make_rom(rom_parameters):

    rom = arby.ReducedOrderModel(**rom_parameters)

    assert rom.Ntrain_ == 101
    assert rom.Nsamples_ == 101

    np.testing.assert_allclose(rom.basis_.data.mean(), 0.136109, rtol=1e-6)
    np.testing.assert_allclose(rom.basis_.data.std(), 1.00563, rtol=1e-6)

    np.testing.assert_allclose(rom.greedy_error_.mean(), 0.00469976, rtol=1e-6)

    np.testing.assert_allclose(
        rom.greedy_error_.std(), 0.01157560317828102, rtol=1e-6
    )

    parameter_interval = rom_parameters["parameter_interval"]
    physical_interval = rom_parameters["physical_interval"]

    to_surrogate = BesselJ(parameter_interval[0], physical_interval)
    surrogate = rom.surrogate(to_surrogate)

    np.testing.assert_allclose(
        surrogate.mean(), 2.726606176467909e-09, rtol=1e-6
    )
    np.testing.assert_allclose(surrogate.std(), 1.357027e-08, rtol=1e-6)


def test_sliced_training(rom_parameters):
    sliced_training = rom_parameters["training_space"][:, :10]
    rom_parameters.update(training_space=sliced_training)
    with pytest.raises(ValueError):
        arby.ReducedOrderModel(**rom_parameters)


@pytest.mark.xfail
def test_wrong_Nsamples(training):
    wrong_Nsamples = 11
    x = np.linspace(0, 1, wrong_Nsamples)
    with pytest.raises(ValueError):
        arby.ReducedOrderModel(training, x)


def test_wrong_Ntrain(rom_parameters):
    rom_parameters.update(parameter_interval=np.linspace(0, 10, 11))

    with pytest.raises(ValueError):
        arby.ReducedOrderModel(**rom_parameters)


@pytest.mark.xfail
def test_alter_Nsamples(training):
    x = np.linspace(0, 1, 101)
    bessel = arby.ReducedOrderModel(training, x)
    bessel.Nsamples_ += 1
    with pytest.raises(ValueError):
        bessel.basis


@pytest.mark.xfail
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


@pytest.mark.xfail
def test_gram_schmidt():
    """Test Gram Schmidt orthonormalization algorithm."""
    expected_basis = np.loadtxt("tests/bessel/bessel_basis.txt")

    x = np.linspace(0, 1, 101)
    integration = arby.Integration(interval=x, rule="riemann")
    computed_basis = arby.gram_schmidt(expected_basis, integration)

    np.testing.assert_allclose(
        computed_basis, expected_basis, rtol=1e-5, atol=1e-8
    )


def test_gram_schmidt_linear_independence():
    expected_basis = np.loadtxt("tests/bessel/bessel_basis.txt")

    x = np.linspace(0, 1, 101)
    integration = arby.Integration(interval=x, rule="riemann")

    non_li_functions = expected_basis
    non_li_functions[0] = expected_basis[1]

    # non linear independence error capturing
    with pytest.raises(ValueError):
        arby.gram_schmidt(expected_basis, integration)

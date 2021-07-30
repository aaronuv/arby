# test_rom.py

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE


import arby

import numpy as np

import pytest

from scipy.special import jv as BesselJ


# =============================================================================
# TESTS
# =============================================================================


def test_wrong_Nsamples(rom_parameters):
    """Test input consistency."""
    rom_parameters.update(physical_points=np.linspace(0, 1, 11))
    with pytest.raises(ValueError):
        arby.ReducedOrderModel(**rom_parameters)


def test_wrong_Ntrain(rom_parameters):
    """Test input consistency."""
    rom_parameters.update(parameter_points=np.linspace(0, 10, 11))
    with pytest.raises(ValueError):
        arby.ReducedOrderModel(**rom_parameters)


def test_rom_rb_interface(rom_parameters):
    """Test API consistency."""
    training_set = rom_parameters["training_set"]
    physical_points = rom_parameters["physical_points"]
    parameter_points = rom_parameters["parameter_points"]

    bessel = arby.ReducedOrderModel(
        training_set, physical_points, parameter_points, greedy_tol=1e-14
    )
    basis = bessel.basis_.data
    errors = bessel.greedy_errors_
    projection_matrix = bessel.projection_matrix_
    greedy_indices = bessel.greedy_indices_
    eim = bessel.eim_

    assert len(basis) == 10
    assert len(errors) == 10
    assert len(projection_matrix) == 101
    assert len(greedy_indices) == 10
    assert eim == bessel.basis_.eim_


def test_surrogate_accuracy():
    """Test surrogate accuracy for Bessel functions."""

    parameter_points = np.linspace(1, 10, num=101)
    nu_validation = np.linspace(1, 10, num=1001)
    physical_points = np.linspace(0, 1, 1001)

    # build training space
    training = np.array(
        [BesselJ(nn, physical_points) for nn in parameter_points]
    )

    # build reduced basis
    bessel = arby.ReducedOrderModel(
        training_set=training,
        physical_points=physical_points,
        parameter_points=parameter_points,
        greedy_tol=1e-15,
    )
    bessel_test = [BesselJ(nn, physical_points) for nn in nu_validation]
    bessel_surrogate = [bessel.surrogate(nn) for nn in nu_validation]

    np.testing.assert_allclose(
        bessel_test, bessel_surrogate, rtol=1e-5, atol=1e-5
    )

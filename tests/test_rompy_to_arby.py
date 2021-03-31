# test_rompy_to_arby.py

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE


# =============================================================================
# IMPORTS
# =============================================================================

import arby

import numpy as np


# =============================================================================
# TESTING
# =============================================================================


def test_regression_reduced_basis(basis_data, training_space):
    "Test that the reduced basis matches ROMpy's for the same "
    "training data"
    physical_interval = np.linspace(0, 1, basis_data.shape[-1])

    # build reduced basis
    basis, _ = arby.reduce_basis(
        training_space, physical_interval, greedy_tol=1e-14
    )

    # compare
    np.testing.assert_allclose(basis.data, basis_data, rtol=1e-5, atol=1e-8)


def test_regression_EIM(basis_data, physical_interval):
    "Test that EIM matches ROMpy's for the same training data"
    rompy_eim_nodes = np.array([0, 100, 2, 36, 9, 72, 1, 20, 89, 4])

    integration = arby.Integration(physical_interval)
    basis = arby.Basis(basis_data, integration)

    np.testing.assert_array_equal(rompy_eim_nodes, basis.eim_.nodes)

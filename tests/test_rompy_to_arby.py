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


def test_regression_reduced_basis(basis, training):
    "Test that the reduced basis matches ROMpy's for the same "
    "training data"
    nbasis, npoints = basis.shape
    physical_inteval = np.linspace(0, 1, npoints)

    # build reduced basis
    rb_bessel = arby.ReducedOrderModel(
        training, physical_inteval, greedy_tol=1e-14
    )
    # compare
    np.testing.assert_allclose(rb_bessel.basis, basis, rtol=1e-5, atol=1e-8)


def test_regression_EIM(
    basis,
):
    "Test that EIM matches ROMpy's for the same training data"
    rompy_eim_nodes = np.array([0, 100, 2, 36, 9, 72, 1, 20, 89, 4])
    # Compute eim nodes for Bessel functions
    eim_bessel = arby.ReducedOrderModel(basis=basis)
    eim_bessel.build_eim()
    # compare
    np.testing.assert_array_equal(rompy_eim_nodes, eim_bessel.eim_nodes_)

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

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.parametrize("rule", ["fake_rule", 1 / 137.0])
def test_bad_integration_inputs(rule):
    """Test rule input."""
    interval = np.linspace(0, 1, 101)
    with pytest.raises(ValueError):
        arby.integrals.Integration(interval=interval, rule=rule)


def test_trapezoidal():
    """Test integration rule."""
    num = 101
    interval = np.linspace(1, 5, num=num)
    function = np.array([5] * num)
    integration = arby.Integration(interval=interval, rule="trapezoidal")
    computed_area_under_curve = integration.integral(function)
    exact_area_under_curve = 20

    np.testing.assert_allclose(
        computed_area_under_curve,
        exact_area_under_curve,
        rtol=1e-5,
        atol=1e-8,
    )


def test_euclidean():
    """Test discrete rule."""
    discrete_points = np.arange(1, 10)
    random = np.random.default_rng(seed=1)
    dummy_array_1 = np.array(
        [random.random() + 1j * random.random() for _ in range(9)]
    )
    dummy_array_2 = np.array(
        [random.random() + 1j * random.random() for _ in range(9)]
    )
    discrete_quadrature = arby.Integration(
        interval=discrete_points, rule="euclidean"
    )
    exact_dot_product = np.dot(dummy_array_1.conjugate(), dummy_array_2)
    computed_dot_product = discrete_quadrature.dot(
        dummy_array_1, dummy_array_2
    )
    np.testing.assert_allclose(
        exact_dot_product, computed_dot_product, atol=1e-6
    )

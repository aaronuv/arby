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

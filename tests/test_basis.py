# test_basis.py

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


def test_wrong_Nsamples(rom_parameters):
    """Test input consistency."""
    rom_parameters.update(physical_points=np.linspace(0, 1, 11))
    with pytest.raises(ValueError):
        arby.reduced_basis(
            training_set=rom_parameters["training_set"],
            physical_points=rom_parameters["physical_points"],
        )


def test_basis_identity(training_set, physical_points, basis_data):
    """Test that computed basis matches stored basis for Bessel data."""
    RB = arby.reduced_basis(training_set, physical_points, greedy_tol=1e-14)
    assert len(RB.basis.data) == 10
    np.testing.assert_allclose(RB.basis.data.mean(), 0.126264, atol=1e-6)
    np.testing.assert_allclose(RB.basis.data.std(), 1.012042, atol=1e-6)
    np.testing.assert_allclose(RB.basis.data, basis_data, atol=1e-12)


def test_basis_shape(basis_data, physical_points):
    """Test that bases shapes are equal for Bessel data."""
    integration = arby.Integration(physical_points)
    basis = arby.Basis(basis_data, integration)

    assert basis.Nbasis_ == len(basis_data)


def test_eim(basis_data, physical_points):
    """Test that computed EIM data matches stored EIM for Bessel data."""
    integration = arby.Integration(physical_points)
    basis = arby.Basis(basis_data, integration)

    np.testing.assert_allclose(basis.eim_.interpolant.mean(), 0.1)
    np.testing.assert_allclose(basis.eim_.interpolant.std(), 0.345161095)
    np.testing.assert_array_equal(
        basis.eim_.nodes, [0, 100, 2, 36, 9, 72, 1, 20, 89, 4]
    )


def test_projection_error(basis_data, physical_points):
    """Test projection_error for dummy points."""
    integration = arby.Integration(physical_points)
    basis = arby.Basis(basis_data, integration)

    perror = basis.projection_error(physical_points)
    np.testing.assert_almost_equal(perror, 0, decimal=10)


def test_project(basis_data, physical_points):
    """Test project method for dummy points."""
    integration = arby.Integration(physical_points)
    basis = arby.Basis(basis_data, integration)

    project = basis.project(physical_points)
    np.testing.assert_allclose(project, physical_points, rtol=1e-4, atol=1e-8)


def test_interpolate(basis_data, physical_points):
    """Test interpolate method for dummy points."""
    integration = arby.Integration(physical_points)
    basis = arby.Basis(basis_data, integration)

    interpolation = basis.interpolate(physical_points)
    np.testing.assert_allclose(
        interpolation, physical_points, rtol=1e-4, atol=1e-8
    )


def test_reduce_basis(training_set):
    physical_points = np.linspace(0, 1, 101)

    RB = arby.reduced_basis(training_set, physical_points, greedy_tol=1e-14)
    basis = RB.basis
    errors = RB.errors
    projection_matrix = RB.projection_matrix
    np.testing.assert_allclose(projection_matrix.mean(), 0.009515, atol=1e-6)
    np.testing.assert_allclose(projection_matrix.std(), 0.061853, atol=1e-6)

    assert basis.eim_.nodes == [0, 100, 2, 36, 9, 72, 1, 20, 89, 4]

    assert len(basis.eim_.interpolant) == 101
    np.testing.assert_allclose(
        basis.eim_.interpolant.mean(), 0.100000, atol=1e-6
    )
    np.testing.assert_allclose(
        basis.eim_.interpolant.std(), 0.345161, atol=1e-6
    )

    np.testing.assert_allclose(errors.mean(), 0.004230, atol=1e-6)
    np.testing.assert_allclose(errors.std(), 0.011072, atol=1e-6)


def test_projector():
    """Test that project method works as true projectors."""
    random = np.random.default_rng(seed=42)

    nu_train = np.linspace(1, 10, num=101)
    physical_points = np.linspace(0, 1, 1001)

    training = np.array([BesselJ(nn, physical_points) for nn in nu_train])

    RB = arby.reduced_basis(training, physical_points, greedy_tol=1e-12)
    basis = RB.basis
    # compute a random index to test Proj_operator^2 = Proj_operator
    sample = random.choice(training)
    proj_fun = basis.project(sample)
    re_proj_fun = basis.project(proj_fun)

    np.testing.assert_allclose(proj_fun, re_proj_fun, rtol=1e-5, atol=1e-8)


def test_interpolator():
    """Test that interpolate method works as true projectors."""
    random = np.random.default_rng(seed=42)

    nu_train = np.linspace(1, 10, num=101)
    physical_points = np.linspace(0, 1, 1001)

    training = np.array([BesselJ(nn, physical_points) for nn in nu_train])

    RB = arby.reduced_basis(training, physical_points, greedy_tol=1e-12)
    basis = RB.basis

    # compute a random index to test Proj_operator^2 = Proj_operator
    sample = random.choice(training)

    interp_fun = basis.interpolate(sample)
    re_interp_fun = basis.interpolate(interp_fun)
    np.testing.assert_allclose(interp_fun, re_interp_fun, rtol=1e-5, atol=1e-8)


def test_projection_error_consistency():
    """Test auto-consistency for projection error function."""

    nu_train = np.linspace(1, 10, num=101)
    physical_points = np.linspace(0, 1, 1001)

    # build traning space
    training = np.array([BesselJ(nn, physical_points) for nn in nu_train])

    # build reduced basis
    RB = arby.reduced_basis(training, physical_points, greedy_tol=1e-12)
    basis = RB.basis
    # Check that projection errors of basis elements onto the basis is
    # zero
    computed_errors = [
        basis.projection_error(basis_element) for basis_element in basis.data
    ]
    expected_errors = [0.0] * basis.Nbasis_
    np.testing.assert_allclose(
        computed_errors, expected_errors, rtol=1e-5, atol=1e-8
    )


@pytest.mark.parametrize("normalize", [True, False])
def test_greedy_already_selected(normalize):
    """Test greedy stopping condition."""

    # Sample parameter nu and physical variable x
    parameter_points = np.linspace(0, 10, num=101)
    physical_points = np.linspace(0, 1, 101)

    # build traning space
    training_set = np.array(
        [BesselJ(nn, physical_points) for nn in parameter_points]
    )

    # build reduced basis with exagerated greedy_tol
    rb_data = arby.reduced_basis(
        training_set, physical_points, greedy_tol=1e-40, normalize=normalize
    )

    assert rb_data.errors[-1] < 1e-15


def test_gram_schmidt(basis_data):
    """Test Gram Schmidt orthonormalization algorithm."""

    physical_points = np.linspace(0, 1, 101)
    integration = arby.Integration(interval=physical_points, rule="riemann")
    computed_basis = arby.gram_schmidt(basis_data, integration)

    np.testing.assert_allclose(
        computed_basis, basis_data, rtol=1e-5, atol=1e-8
    )


def test_gram_schmidt_linear_independence(basis_data):

    x = np.linspace(0, 1, 101)
    integration = arby.Integration(interval=x, rule="riemann")

    non_li_functions = basis_data[:]
    non_li_functions[0] = basis_data[1]

    # non linear independence error capturing
    with pytest.raises(ValueError):
        arby.gram_schmidt(basis_data, integration)


@pytest.mark.parametrize("rule", ["euclidean", "riemann", "trapezoidal"])
@pytest.mark.parametrize("normalize", [True, False])
def test_linear_model(normalize, rule):
    """Test that a linear model gives one-element basis."""
    nu = np.linspace(1, 5, 101)
    x = np.linspace(1, 2, 101)

    # create a training set for f(nu, x) = nu * x^2
    training = np.array([nu * x ** 2 for nu in nu])
    rb_data = arby.reduced_basis(
        training_set=training,
        physical_points=x,
        integration_rule=rule,
        greedy_tol=1e-15,
        normalize=normalize,
    )

    expected_dim = 1
    assert len(rb_data.indices) == expected_dim
    assert rb_data.basis.Nbasis_ == expected_dim
    assert rb_data.errors.size == expected_dim
    assert rb_data.projection_matrix.shape[1] == expected_dim


@pytest.mark.parametrize("rule", ["euclidean", "riemann", "trapezoidal"])
@pytest.mark.parametrize("normalize", [True, False])
def test_2dim_model(normalize, rule):
    """Test that 2-dim model gives a two-element basis."""
    nu = np.linspace(1, 2, 101)
    x = np.linspace(1, 2, 101)

    # create a training set for f(nu, x) = sin(nu) * x^2 + nu^2 * x^3
    training = np.array([np.sin(nu) * x ** 2 + nu ** 2 * x ** 3 for nu in nu])
    rb_data = arby.reduced_basis(
        training_set=training,
        physical_points=x,
        integration_rule=rule,
        greedy_tol=1e-15,
        normalize=normalize,
    )

    expected_dim = 2
    assert len(rb_data.indices) == expected_dim
    assert rb_data.basis.Nbasis_ == expected_dim
    assert rb_data.errors.size == expected_dim
    assert rb_data.projection_matrix.shape[1] == expected_dim


def test_zero_seed_selected(rom_parameters):
    """Test non-zero seed selection."""
    training_set = rom_parameters["training_set"]
    # replace seed by a null vector
    # the next series is a non-zero vector
    training_set[0] = np.zeros(training_set.shape[1])
    rb_data = arby.reduced_basis(
        training_set=training_set,
        physical_points=rom_parameters["physical_points"],
    )
    # check the next seed is indeed the first basis element
    assert rb_data.indices[0] == 1


def test_null_training(rom_parameters):
    """Test null training set detection."""
    rom_parameters.update(
        training_set=np.zeros_like(rom_parameters["training_set"])
    )

    with pytest.raises(ValueError):
        arby.reduced_basis(
            training_set=rom_parameters["training_set"],
            physical_points=rom_parameters["physical_points"],
        )

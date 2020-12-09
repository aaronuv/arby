# test_arby.py

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE

import unittest
from random import randint

import arby

import numpy as np

from scipy.special import jv as BesselJ


class TestArby_core(unittest.TestCase):
    def test_basis_shape(self):
        """Test correct shape for reduced basis."""
        npoints = 101
        # Sample parameter nu and physical variable x
        nu = np.linspace(0, 10, num=npoints)
        x = np.linspace(0, 1, 101)
        # build traning space
        training = np.array([BesselJ(nn, x) for nn in nu])
        # build reduced basis
        bessel = arby.ReducedOrderModeling(training, x, greedy_tol=1e-12)

        # Assert that basis has correct shape
        self.assertEqual(bessel.basis.ndim, 2)
        self.assertEqual(bessel.basis.shape[1], npoints)

    def test_surrogate(self):
        """Test surrogate accuracy."""
        npoints = 101
        nu_train = np.linspace(1, 10, num=npoints)
        nu_validation = np.linspace(1, 10, num=1001)
        x = np.linspace(0, 1, 1001)
        # build traning space
        training = np.array([BesselJ(nn, x) for nn in nu_train])
        # build reduced basis
        bessel = arby.ReducedOrderModeling(
            training_space=training,
            physical_interval=x,
            parameter_interval=nu_train,
            greedy_tol=1e-15,
        )
        bessel_test = [BesselJ(nn, x) for nn in nu_validation]
        bessel_surrogate = [bessel.surrogate(nn) for nn in nu_validation]
        self.assertTrue(
            np.allclose(bessel_test, bessel_surrogate, rtol=1e-5, atol=1e-5)
        )

    def test_gram_schmidt(self):
        """Test Gram Schmidt orthonormalization algorithm."""
        expected_basis = np.loadtxt("tests/bessel/bessel_basis.txt")
        nbasis, npoints = expected_basis.shape
        x = np.linspace(0, 1, 101)
        integration = arby.Integration(interval=x, rule="riemann")
        computed_basis = arby.gram_schmidt(expected_basis, integration)
        self.assertTrue(
            np.allclose(computed_basis, expected_basis, rtol=1e-5, atol=1e-8)
        )

    def test_projectors(self):
        """Test that projectors works as true projectors."""
        npoints = 101
        nu_train = np.linspace(1, 10, num=npoints)
        x = np.linspace(0, 1, 1001)
        # build traning space
        training = np.array([BesselJ(nn, x) for nn in nu_train])
        # build reduced basis
        bessel = arby.ReducedOrderModeling(
            training_space=training,
            physical_interval=x,
            parameter_interval=nu_train,
            greedy_tol=1e-12)
        # compute a random index to test Proj_operator^2 = Proj_operator
        random_index = randint(0, npoints)
        proj_fun = bessel.project(training[random_index], bessel.basis)
        proj2_fun = bessel.project(proj_fun, bessel.basis)
        self.assertTrue(
            np.allclose(proj_fun, proj2_fun, rtol=1e-5, atol=1e-8)
        )
        # compute a random index to test Interpolant^2 = Interpolant
        random_index = randint(0, npoints)
        # build interpolant matrix
        bessel.build_eim()
        interp_fun = bessel.interpolate(training[random_index])
        interp2_fun = bessel.interpolate(interp_fun)
        self.assertTrue(
            np.allclose(interp_fun, interp2_fun, rtol=1e-5, atol=1e-8))


class TestArby_Integrals(unittest.TestCase):
    def test_Integration_inputs(self):
        """Test rule input."""
        with self.assertRaises(ValueError):
            interval = np.linspace(0, 1, 101)
            rule = "fake_rule"
            arby.integrals.Integration(interval=interval, rule=rule)

    def test_Integration_trapezoidal(self):
        """Test integration rule."""
        num = 101
        interval = np.linspace(1, 5, num=num)
        function = np.array([5] * num)
        integration = arby.Integration(interval=interval, rule="trapezoidal")
        computed_area_under_curve = integration.integral(function)
        exact_area_under_curve = 20
        self.assertTrue(
            np.allclose(
                computed_area_under_curve,
                exact_area_under_curve,
                rtol=1e-5,
                atol=1e-8,
            )
        )


if __name__ == "__main__":
    unittest.main()

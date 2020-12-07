# test_arby.py

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE

import unittest

import arby

import numpy as np

from scipy.special import jv as BesselJ


class TestArby(unittest.TestCase):
    def test_basis_shape(self):
        "Test correct shape for reduced basis"
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
        "Test surrogate accuracy"
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

    def test_GramSchmidt(self):
        "Test Gram Schmidt orthonormalization algorithm"
        expected_basis = np.loadtxt("tests/bessel/bessel_basis.txt")
        nbasis, npoints = expected_basis.shape
        x = np.linspace(0, 1, 101)
        integration = arby.Integration(interval=x, rule="riemann")
        computed_basis = arby.gram_schmidt(expected_basis, integration)
        self.assertTrue(
            np.allclose(computed_basis, expected_basis, rtol=1e-5, atol=1e-8)
        )


class TestIntegrals(unittest.TestCase):
    def test_Integration_inputs(self):
        with self.assertRaises(ValueError):
            interval = np.linspace(0, 1, 101)
            rule = "fake_rule"
            arby.integrals.Integration(interval=interval, rule=rule)

    def test_Integration_trapezoidal(self):
        "Test integration rule"
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

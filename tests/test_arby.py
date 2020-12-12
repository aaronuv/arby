# test_arby.py

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE

import unittest

import arby

import numpy as np

from scipy.special import jv as BesselJ


class TestArby_core(unittest.TestCase):
    def test_inputs(self):
        training = np.loadtxt("tests/bessel/bessel_training.txt")
        with self.assertRaises(ValueError):
            Nsamples = 10
            x = np.linspace(0, 1, 101)
            sliced_training = training[:, :Nsamples]
            arby.ReducedOrderModel(sliced_training, x)

        with self.assertRaises(ValueError):
            Nsamples = training.shape[1]
            wrong_Nsamples = 11
            x = np.linspace(0, 1, wrong_Nsamples)
            arby.ReducedOrderModel(training, x)

        with self.assertRaises(ValueError):
            wrong_Ntrain = 11
            x = np.linspace(0, 1, 101)
            nu = np.linspace(0, 10, wrong_Ntrain)
            arby.ReducedOrderModel(training, x, nu)

    def test_basis_shape(self):
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
        self.assertEqual(bessel.basis.ndim, 2)
        self.assertEqual(bessel.basis.shape[1], npoints)

    def test_greedy_already_selected(self):
        """Test greedy stopping condition."""
        npoints = 101
        # Sample parameter nu and physical variable x
        nu = np.linspace(0, 10, num=npoints)
        x = np.linspace(0, 1, 101)
        # build traning space
        training = np.array([BesselJ(nn, x) for nn in nu])
        # build reduced basis with exagerated greedy_tol
        bessel = arby.ReducedOrderModel(training, x, greedy_tol=1e-20)
        bessel.basis

    def test_surrogate(self):
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
        # non linear independence error capturing
        with self.assertRaises(ValueError):
            non_li_functions = expected_basis
            non_li_functions[0] = expected_basis[1]
            computed_basis = arby.gram_schmidt(expected_basis, integration)

    def test_projectors(self):
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
        self.assertTrue(np.allclose(proj_fun, proj2_fun, rtol=1e-5, atol=1e-8))
        # compute a random index to test Interpolant^2 = Interpolant
        random_index = np.random.randint(0, npoints)
        # build interpolant matrix
        bessel.build_eim()
        interp_fun = bessel.interpolate(training[random_index])
        interp2_fun = bessel.interpolate(interp_fun)
        self.assertTrue(
            np.allclose(interp_fun, interp2_fun, rtol=1e-5, atol=1e-8)
        )

    def test_projection_error(self):
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
        self.assertTrue(
            np.allclose(computed_errors, expected_errors, rtol=1e-5, atol=1e-8)
        )


class TestArby_Integrals(unittest.TestCase):
    def test_Integration_inputs(self):
        """Test rule input."""
        with self.assertRaises(ValueError):
            interval = np.linspace(0, 1, 101)
            rule = "fake_rule"
            arby.integrals.Integration(interval=interval, rule=rule)
        with self.assertRaises(ValueError):
            interval = None
            rule = "riemann"
            arby.integrals.Integration(interval=interval, rule=rule)
        with self.assertRaises(TypeError):
            interval = np.linspace(0, 1, 101)
            rule = 1 / 137
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

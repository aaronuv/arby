import unittest
import arby
import numpy as np


class TestArby(unittest.TestCase):
    def test_basis_shape(self):
        "Test correct shape for reduced basis"
        from scipy.special import jv as BesselJ

        npoints = 101
        # Sample parameter nu and physical variable x
        nu = np.linspace(0, 10, num=npoints)
        x = np.linspace(0, 1, 101)
        # build traning space
        training = np.array([BesselJ(nn, x) for nn in nu])
        # build reduced basis
        rb = arby.ReducedOrderModeling(training, x, greedy_tol=1e-14)

        # Assert that basis has correct shape
        self.assertEqual(rb.basis.ndim, 2)
        self.assertEqual(rb.basis.shape[1], npoints)

    def test_GramSchmidt(self):
        expected_basis = np.loadtxt("tests/bessel/bessel_basis.txt")
        nbasis, npoints = expected_basis.shape
        integration = arby.Integration([0, 1], num=npoints, rule="riemann")
        computed_basis = arby.GramSchmidt(expected_basis, integration)
        self.assertTrue(
            np.allclose(computed_basis, expected_basis, rtol=1e-5, atol=1e-8)
        )


class TestIntegrals(unittest.TestCase):
    def test_Integration_inputs(self):

        with self.assertRaises(TypeError):
            rule_test = 1.0
            npoints = 101
            arby.integrals.Integration([0, 1], num=npoints, rule=rule_test)
        with self.assertRaises(ValueError):
            arby.integrals.Integration([0, 1], rule="riemann")
        with self.assertRaises(ValueError):
            npoints = 101
            rule = "fake_rule"
            arby.integrals.Integration([0, 1], num=npoints, rule=rule)


if __name__ == "__main__":
    unittest.main()

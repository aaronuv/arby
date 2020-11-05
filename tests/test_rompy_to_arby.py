import unittest
import arby
import numpy as np


class TestBesselExample(unittest.TestCase):
    def setUp(self):
        self.basis = np.loadtxt("tests/bessel/bessel_basis.txt")
        self.training = np.loadtxt("tests/bessel/bessel_training.txt")

    def test_regression_reduced_basis(self):
        "Test that the reduced basis matches ROMpy's for the same training data"

        nbasis, npoints = self.basis.shape
        # set integration rule
        integration = arby.integrals.Integration(
            [0, 1], num=npoints, rule="riemann"
        )
        # build reduced basis
        rb = arby.greedy.ReducedBasis(integration)
        rb.make(self.training, 0, 1e-14, verbose=False)
        # compare
        self.assertTrue(
            np.allclose(rb.basis, self.basis, rtol=1e-5, atol=1e-8)
        )

    def test_regression_EIM(self):
        "Test that EIM matches ROMpy's for the same training data"
        
        rompy_eim_nodes = np.array([0, 100, 2, 36, 9, 72, 1, 20, 89, 4])
        # Compute eim nodes for Bessel functions
        eim_bessel = arby.eim.EmpiricalMethods(self.basis)
        eim_bessel.eim()
        # compare
        self.assertTrue(
            (rompy_eim_nodes == eim_bessel.indices).all()
        )

if __name__ == "__main__":
    unittest.main()

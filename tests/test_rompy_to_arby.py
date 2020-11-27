import unittest
import arby
import numpy as np


class TestBesselExample(unittest.TestCase):
    def setUp(self):
        self.basis = np.loadtxt("tests/bessel/bessel_basis.txt")
        self.training = np.loadtxt("tests/bessel/bessel_training.txt")

    def test_regression_reduced_basis(self):
        "Test that the reduced basis matches ROMpy's for the same "
        "training data"
        nbasis, npoints = self.basis.shape
        physical_inteval = np.linspace(0, 1, npoints)
        # build reduced basis
        rb_bessel = arby.ReducedOrderModeling(self.training, physical_inteval)
        rb_bessel.build_reduced_basis(tol=1e-12)
        # compare
        self.assertTrue(np.allclose(rb_bessel.basis, self.basis,
                                    rtol=1e-5, atol=1e-8))

    def test_regression_EIM(self):
        "Test that EIM matches ROMpy's for the same training data"
        rompy_eim_nodes = np.array([0, 100, 2, 36, 9, 72, 1, 20, 89, 4])
        # Compute eim nodes for Bessel functions
        eim_bessel = arby.ReducedOrderModeling(basis=self.basis)
        eim_bessel.build_eim()
        # compare
        self.assertTrue((rompy_eim_nodes == eim_bessel.eim_nodes).all())


if __name__ == "__main__":
    unittest.main()

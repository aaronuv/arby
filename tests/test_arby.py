import unittest
import arby
import numpy as np


class TestArby(unittest.TestCase):
    def test_basis_shape(self):
        "Test correct shape for reduced basis"
        from scipy.special import jv as BesselJ

        npoints = 101
        # Sample parameter nu between 0 and 10
        nu = np.linspace(0, 10, num=npoints)
        # set integration rule
        integration = arby.integrals.Integration([0, 1], num=npoints,
                                                 rule="riemann")
        # build traning space
        training = np.array([BesselJ(nn, integration.nodes) for nn in nu])
        # build reduced basis
        rb = arby.greedy.ReducedBasis(integration)
        rb.make(training, 0, 1e-14, verbose=False)

        # Assert that basis has correct shape
        self.assertEqual(rb.basis.ndim, 2)
        self.assertEqual(rb.basis.shape[1], npoints)

if __name__ == "__main__":
    unittest.main()

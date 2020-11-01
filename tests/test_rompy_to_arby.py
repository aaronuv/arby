import unittest
import arby
import numpy as np
from scipy.special import jv as BesselJ

class TestBesselExample(unittest.TestCase):
    def setUp(self):
        self.basis = np.loadtxt("tests/bessel/bessel_basis.txt")
        self.training = np.loadtxt("tests/bessel/bessel_training.txt")

    def test_bessel_consistent(self):
        self.assertEqual(self.basis.shape, (10, 101))
        self.assertEqual(self.training.shape, (101, 101))


    def test_bessel_regression(self):
        nu = np.linspace(0, 10, num=101)
        #set integration rule
        integration = arby.integrals.Integration([0,1], num=101, rule='riemann')
        #set integration nodes
        x = integration.nodes
        #build traning space
        training = np.array([BesselJ(nn, x) for nn in nu])
        #build reduced basis
        rb = arby.greedy.ReducedBasis(integration)
        rb.make(training, 0, 1e-14, verbose=True)
        #compare
        #atol=1e-8 and rtol=1e-5 are default values
        np.allclose(rb.basis,self.basis)
        
if __name__ == "__main__":
    unittest.main()

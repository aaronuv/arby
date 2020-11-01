import unittest
import arby
import numpy as np
from scipy.special import jv as BesselJ

class TestBesselExample(unittest.TestCase):
    def setUp(self):
        self.basis = np.loadtxt("tests/bessel/bessel_basis.txt")
        self.training = np.loadtxt("tests/bessel/bessel_training.txt")

    def test_bessel_regression(self):
#        nu = np.linspace(0, 10, num=101)
        #set integration rule
        integration = arby.integrals.Integration([0,1], num=101, rule='riemann')
        #set integration nodes
#        x = integration.nodes
        #build traning space
#        training = np.array([BesselJ(nn, x) for nn in nu])
        training = self.training
        #build reduced basis
        rb = arby.greedy.ReducedBasis(integration)
        rb.make(training, 0, 1e-14, verbose=False)
        #compare
        self.assertTrue(np.allclose(rb.basis,self.basis,rtol=1e-5,atol=1e-8))

        
if __name__ == "__main__":
    unittest.main()

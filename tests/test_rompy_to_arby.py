import unittest
import arby
import numpy as np

class TestBesselExample(unittest.TestCase):
    def setUp(self):
        basis = np.loadtxt("tests/bessel/bessel_basis.txt")
        training = np.loadtxt("tests/bessel/bessel_training.txt")

    def test_bessel_consistent(self):
        self.assertEqual(basis.shape, (10, 101))
        self.assertEqual(training.shape, (101, 101))


if __name__ == "__main__":
    unittest.main()

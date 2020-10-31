import unittest
import arby
import numpy as np

class TestBesselExample(unittest.TestCase):
    def setUp(self):
        self.basis = np.loadtxt("tests/bessel/bessel_basis.txt")
        self.training = np.loadtxt("tests/bessel/bessel_training.txt")

    def test_bessel_consistent(self):
        self.assertEqual(self.basis.shape, (10, 101))
        self.assertEqual(self.training.shape, (101, 101))


if __name__ == "__main__":
    unittest.main()

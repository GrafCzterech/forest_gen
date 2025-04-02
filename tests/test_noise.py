import unittest
from forest_gen.terrain import NOISE_FUNC


class TestNoise(unittest.TestCase):
    def test_0(self):
        self.assertAlmostEqual(NOISE_FUNC(0.0, 0.0), 2.5, 4)

    def test_1(self):
        self.assertAlmostEqual(NOISE_FUNC(1.0, 1.0), 2.0784, 4)

    def test_2(self):
        self.assertAlmostEqual(NOISE_FUNC(2.0, 2.0), 2.10615, 4)

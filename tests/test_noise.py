import unittest
from forest_gen.terrain import NOISE_FUNC


class TestNoise(unittest.TestCase):
    def test_0(self):
        self.assertAlmostEqual(NOISE_FUNC(0.0, 0.0), 20.0, 4)

    def test_1(self):
        self.assertAlmostEqual(NOISE_FUNC(1.0, 1.0), 17.80344787516461, 4)

    def test_2(self):
        self.assertAlmostEqual(NOISE_FUNC(2.0, 2.0), 15.928450969955646, 4)

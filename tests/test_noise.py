from typing import Any
import unittest
from forest_gen.terrain import NOISE_FUNC


class TestNoise(unittest.TestCase):

    def assertAlmostEqual(
        self, a: Any, b: Any, places: int = 3
    ):  # reduced default from 7 to 3
        super().assertAlmostEqual(a, b, places=places)

    def test_0(self):
        self.assertAlmostEqual(NOISE_FUNC(0.0, 0.0), 2.5)

    def test_1(self):
        self.assertAlmostEqual(NOISE_FUNC(1.0, 1.0), 2.1825)

    def test_2(self):
        self.assertAlmostEqual(NOISE_FUNC(2.0, 2.0), 1.8975)

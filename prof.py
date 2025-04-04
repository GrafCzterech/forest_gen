import cProfile

from forest_gen.heightmap import NOISE_FUNC


def test_func():
    for i in range(1000):
        NOISE_FUNC(i, i)


cProfile.run(
    "test_func()",
    sort="time",
)

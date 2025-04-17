import cProfile

from forest_gen.heightmap import NOISE_FUNC
from forest_gen.asset_dist import Simulation, Species


def test_func():
    for i in range(1000):
        NOISE_FUNC(i, i)


def test_sim():
    spec_a = Species("a", 5, 0.1)
    spec_b = Species("b", 10, 0.2)
    spec_c = Species("c", 15, 0.3)

    sim = Simulation((100.0, 100.0), {"tree": {spec_a, spec_b, spec_c}})
    state = sim.new_state(1.0)
    state.run_state(5)


if __name__ == "__main__":
    cProfile.run(
        "test_sim()",
        sort="time",
    )

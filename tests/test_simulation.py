import unittest
from forest_gen.asset_dist import SimulationState, Plant, Species, Simulation
from math import dist


class TestSimulationState(unittest.TestCase):

    species_a = Species("a", 5, 0.1, radius=2.0)
    species_b = Species("b", 4, 0.1)

    def test_iter(self):
        init = (
            Plant((0, 0), self.species_a, 0),
            Plant((1, 1), self.species_b, 0),
        )
        state = SimulationState(init, (10, 10))
        for el in init:
            self.assertIn(el, state)
        state = SimulationState(init, (20, 20))
        for el in init:
            self.assertIn(el, state)

    def test_find_nearby(self):
        init = (
            Plant((0, 0), self.species_a, 0),
            Plant((1, 1), self.species_b, 0),
        )
        state = SimulationState(init, (10, 10))
        self.assertIn(init[1], state.get_nearby(init[0]))
        state = SimulationState(init, (20, 20))
        self.assertIn(init[1], state.get_nearby(init[0]))

    def test_auto_find_nearby(self):
        size = 10
        for size in range(1, 50):
            spec = Species("spec", 5, 0.1, radius=float(size))
            init = (Plant((i, i), spec, 0) for i in range(size))
            state = SimulationState(init, (size, size))
            for el in init:
                self.assertIn(el, state.get_nearby(el))

    def test_init_state(self):
        state = Simulation(
            (100, 100), {"trees": {self.species_a, self.species_b}}
        ).new_state(1.0)
        not_empty = False
        for i, a in enumerate(state):
            not_empty = True
            for j, b in enumerate(state):
                if i == j:
                    continue
                self.assertGreaterEqual(dist(a.coords, b.coords), a.radius)
        self.assertTrue(not_empty)

    def test_post_sim_state(self):
        state = Simulation(
            (100, 100), {"trees": {self.species_a, self.species_b}}
        ).new_state(1.0)
        state.run_state(5)
        not_empty = False
        for i, a in enumerate(state):
            not_empty = True
            for j, b in enumerate(state):
                if i == j:
                    continue
                self.assertGreaterEqual(dist(a.coords, b.coords), a.radius)
        self.assertTrue(not_empty)


if __name__ == "__main__":
    unittest.main()

import math
import pytest
import numpy as np


def _resolve(sym, name: str, *paths: str):
    errs = []
    for p in paths:
        try:
            return sym(p, name)
        except Exception as e:
            errs.append(f"{p}.{name}: {type(e).__name__}({e})")
    raise RuntimeError(
        f"Could not resolve '{name}' from any of: {paths}\n" + "\n".join(errs)
    )


def _pairwise_clearance_ok(plants, min_clearance_fn) -> bool:
    plants = list(plants)
    for i in range(len(plants)):
        for j in range(i + 1, len(plants)):
            a, b = plants[i], plants[j]
            required = min_clearance_fn(a, b)
            if math.dist(a.coords, b.coords) < required - 1e-12:
                return False
    return True


@pytest.mark.unit
def test_new_state_points_in_bounds_and_no_intraspecies_conflicts(sym):
    """
    SciPy PoissonDisk output plants must be within bounds
    and respect minimum spacing implied by the sampler radius.
    """
    pytest.importorskip("scipy")

    Species = _resolve(sym, "Species", "asset_dist.definitions", "asset_dist", "definitions")
    Simulation = _resolve(sym, "Simulation", "asset_dist.sim", "asset_dist")

    size = (30.0, 30.0)
    sp = Species(
        name="A",
        max_age=50,
        radius=2.0,
        species_density=0.01, 
        reproduction_rate=0,
        juvenile_mortality_depth=0.0,
        juvenile_recovery_age=0.2,
        viability_map=lambda x, y: 1.0,  # noqa: E731
    )
    sim = Simulation(size=size, species={"canopy": {sp}})
    state = sim.new_state(scene_density=1.0)

    # Bounds
    for plant in state:
        assert 0.0 <= plant.coords[0] <= size[0]
        assert 0.0 <= plant.coords[1] <= size[1]
        assert plant.species is sp
        assert plant.age == 0

    plants = list(state)
    assert _pairwise_clearance_ok(plants, lambda a, b: max(a.species.radius, b.species.radius))


@pytest.mark.unit
def test_new_state_prioritizes_larger_radius_species_when_conflicts(sym, monkeypatch):
    """
    Regress guard species are sorted by radius descending.
    If two species propose conflicting points, the large-radius species should win.
    """
    pytest.importorskip("scipy")

    Species = _resolve(sym, "Species", "asset_dist.definitions", "asset_dist", "definitions")
    Simulation = _resolve(sym, "Simulation", "asset_dist.sim", "asset_dist")

    sim_mod = __import__(Simulation.__module__, fromlist=["_"])

    class _PoissonStub:
        def __init__(self, d, radius, l_bounds, u_bounds):  # noqa: ARG002
            self.radius = radius

        def random(self, n):
            if self.radius >= 5.0:
                return np.array([[5.0, 5.0]], dtype=float)
            return np.array([[5.5, 5.0]], dtype=float)  # conflicts with big radius

        def reset(self):
            return None

    monkeypatch.setattr(sim_mod, "PoissonDisk", _PoissonStub)

    size = (10.0, 10.0) 
    big = Species(
        "Big",
        max_age=10,
        radius=5.0,
        species_density=0.01, 
        reproduction_rate=0,
        juvenile_mortality_depth=0.0,
        juvenile_recovery_age=0.2,
        viability_map=lambda x, y: 1.0,  # noqa: E731
    )
    small = Species(
        "Small",
        max_age=10,
        radius=1.0,
        species_density=0.01,
        reproduction_rate=0,
        juvenile_mortality_depth=0.0,
        juvenile_recovery_age=0.2,
        viability_map=lambda x, y: 1.0,  # noqa: E731
    )

    sim = Simulation(size=size, species={"all": {big, small}})
    state = sim.new_state(scene_density=1.0)

    plants = list(state)
    assert len(plants) == 1
    assert plants[0].species.name == "Big"


@pytest.mark.unit
def test_new_state_skips_species_with_zero_desired_count(sym, monkeypatch):
    """
    If desired_n floors to 0, the sampler must not be constructed/called.
    This prevents wasted work and accidental creation of plants.
    """
    pytest.importorskip("scipy")

    Species = _resolve(sym, "Species", "asset_dist.definitions", "asset_dist", "definitions")
    Simulation = _resolve(sym, "Simulation", "asset_dist.sim", "asset_dist")

    sim_mod = __import__(Simulation.__module__, fromlist=["_"])

    class _PoissonExplode:
        def __init__(self, *a, **k):  # noqa: ANN001
            raise AssertionError("PoissonDisk should not be constructed when n <= 0")

    monkeypatch.setattr(sim_mod, "PoissonDisk", _PoissonExplode)

    size = (10.0, 10.0)  # area=100
    sp = Species(
        "Zero",
        max_age=10,
        radius=1.0,
        species_density=0.0,  # desired_n = 0 => n=0
        reproduction_rate=0,
        juvenile_mortality_depth=0.0,
        juvenile_recovery_age=0.2,
        viability_map=lambda x, y: 1.0,  # noqa: E731
    )

    sim = Simulation(size=size, species={"x": {sp}})
    state = sim.new_state(scene_density=1.0)
    assert len(list(state)) == 0


@pytest.mark.unit
def test_new_state_uses_clearance_radius_semantics_max_not_sum(sym, monkeypatch):
    """
    Locks in current semantics: 'radius' is a clearance requirement,
    so required distance is max(r1, r2), not (r1 + r2).
    """
    pytest.importorskip("scipy")

    Species = _resolve(sym, "Species", "asset_dist.definitions", "asset_dist", "definitions")
    Simulation = _resolve(sym, "Simulation", "asset_dist.sim", "asset_dist")

    sim_mod = __import__(Simulation.__module__, fromlist=["_"])


    class _PoissonStub:
        def __init__(self, d, radius, l_bounds, u_bounds):  # noqa: ARG002
            self.radius = radius

        def random(self, n):
            if self.radius > 3.0:
                return np.array([[0.0, 0.0]], dtype=float)
            return np.array([[4.0, 0.0]], dtype=float)

        def reset(self):
            return None

    monkeypatch.setattr(sim_mod, "PoissonDisk", _PoissonStub)

    size = (10.0, 10.0)
    a = Species(
        "A",
        max_age=10,
        radius=3.1,
        species_density=0.01,
        reproduction_rate=0,
        juvenile_mortality_depth=0.0,
        juvenile_recovery_age=0.2,
        viability_map=lambda x, y: 1.0,  # noqa: E731
    )
    b = Species(
        "B",
        max_age=10,
        radius=3.0,
        species_density=0.01,
        reproduction_rate=0,
        juvenile_mortality_depth=0.0,
        juvenile_recovery_age=0.2,
        viability_map=lambda x, y: 1.0,  # noqa: E731
    )

    sim = Simulation(size=size, species={"x": {a, b}})
    state = sim.new_state(scene_density=1.0)
    plants = list(state)

    assert {p.species.name for p in plants} == {"A", "B"}
    assert _pairwise_clearance_ok(plants, lambda p, q: max(p.species.radius, q.species.radius))

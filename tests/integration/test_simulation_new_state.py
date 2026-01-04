import math
import pytest


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


def _pairwise_clearance_ok(plants) -> bool:
    plants = list(plants)
    for i in range(len(plants)):
        for j in range(i + 1, len(plants)):
            a, b = plants[i], plants[j]
            required = max(a.species.radius, b.species.radius)  # your semantics in sim.py
            if math.dist(a.coords, b.coords) < required - 1e-12:
                return False
    return True


@pytest.mark.integration
def test_simulation_new_state_multispecies_bounds_and_clearance(sym):
    pytest.importorskip("scipy")

    Species = _resolve(sym, "Species", "asset_dist.definitions", "asset_dist", "definitions")
    Simulation = _resolve(sym, "Simulation", "asset_dist.sim", "asset_dist")

    size = (60.0, 60.0)
    area = size[0] * size[1]

    big = Species(
        name="Big",
        max_age=50,
        radius=4.0,
        species_density=0.0015,  # ~5 on 3600 m2
        reproduction_rate=0,
        juvenile_mortality_depth=0.0,
        juvenile_recovery_age=0.2,
        viability_map=lambda x, y: 1.0,  # noqa: E731
    )
    small = Species(
        name="Small",
        max_age=50,
        radius=1.2,
        species_density=0.004,   # ~14
        reproduction_rate=0,
        juvenile_mortality_depth=0.0,
        juvenile_recovery_age=0.2,
        viability_map=lambda x, y: 1.0,  # noqa: E731
    )

    sim = Simulation(size=size, species={"canopy": {big}, "understory": {small}})
    state = sim.new_state(scene_density=1.0)

    plants = list(state)
    assert len(plants) > 0

    # Bounds
    for p in plants:
        assert 0.0 <= p.coords[0] <= size[0]
        assert 0.0 <= p.coords[1] <= size[1]
        assert p.age == 0
        assert p.species.name in {"Big", "Small"}

    # Clearance invariant (your current semantics is max(r1, r2))
    assert _pairwise_clearance_ok(plants)


@pytest.mark.integration
def test_simulation_new_state_contains_both_species_when_density_is_reasonable(sym):
    pytest.importorskip("scipy")

    Species = _resolve(sym, "Species", "asset_dist.definitions", "asset_dist", "definitions")
    Simulation = _resolve(sym, "Simulation", "asset_dist.sim", "asset_dist")

    size = (80.0, 80.0)

    big = Species(
        name="Big",
        max_age=50,
        radius=4.5,
        species_density=0.001,   # ~6
        reproduction_rate=0,
        juvenile_mortality_depth=0.0,
        juvenile_recovery_age=0.2,
        viability_map=lambda x, y: 1.0,  # noqa: E731
    )
    small = Species(
        name="Small",
        max_age=50,
        radius=1.0,
        species_density=0.0025,  # ~16
        reproduction_rate=0,
        juvenile_mortality_depth=0.0,
        juvenile_recovery_age=0.2,
        viability_map=lambda x, y: 1.0,  # noqa: E731
    )

    sim = Simulation(size=size, species={"a": {big}, "b": {small}})
    state = sim.new_state(scene_density=1.0)

    kinds = {p.species.name for p in state}

    # We avoid asserting exact counts (PoissonDisk is stochastic),
    # but both species should normally appear at these densities.
    assert "Big" in kinds
    assert "Small" in kinds


@pytest.mark.integration
def test_simulation_new_state_large_species_not_starved_by_small(sym):
    """
    Regression guard: the algorithm sorts species by radius descending.
    If someone removes that sort, large species often get starved by small ones.
    """
    pytest.importorskip("scipy")

    Species = _resolve(sym, "Species", "asset_dist.definitions", "asset_dist", "definitions")
    Simulation = _resolve(sym, "Simulation", "asset_dist.sim", "asset_dist")

    size = (50.0, 50.0)

    big = Species(
        name="Big",
        max_age=50,
        radius=5.0,
        species_density=0.0016,  # ~4
        reproduction_rate=0,
        juvenile_mortality_depth=0.0,
        juvenile_recovery_age=0.2,
        viability_map=lambda x, y: 1.0,  # noqa: E731
    )
    small = Species(
        name="Small",
        max_age=50,
        radius=1.0,
        species_density=0.01,    # ~25, could starve Big if placed first
        reproduction_rate=0,
        juvenile_mortality_depth=0.0,
        juvenile_recovery_age=0.2,
        viability_map=lambda x, y: 1.0,  # noqa: E731
    )

    sim = Simulation(size=size, species={"a": {big}, "b": {small}})
    state = sim.new_state(scene_density=1.0)

    big_count = sum(1 for p in state if p.species.name == "Big")
    small_count = sum(1 for p in state if p.species.name == "Small")

    # We don't demand exact counts, just that Big isn't wiped out.
    assert big_count >= 1
    assert small_count >= 1

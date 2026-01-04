import random
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


def _species(Species, *, name="S", max_age=50, radius=0.05, reproduction_rate=5, reproduction_radius=2.0, species_density=0.1):
    #  vt() predictable
    return Species(
        name=name,
        max_age=max_age,
        radius=radius,
        reproduction_rate=reproduction_rate,
        reproduction_radius=reproduction_radius,
        species_density=species_density,
        juvenile_mortality_depth=0.0,
        juvenile_recovery_age=0.2,
        viability_map=lambda x, y: 1.0,  # noqa: E731
    )


@pytest.mark.integration
def test_run_state_ages_and_spawns_in_bounds(sym):
    """
    Integration: run_state should
    - age existing plants,
    - spawn new plants (with deterministic seed),
    - keep everything within bounds.
    """
    Species = _resolve(sym, "Species", "asset_dist.definitions", "asset_dist", "definitions")
    Plant = _resolve(sym, "Plant", "asset_dist.definitions", "asset_dist", "definitions")
    SimulationState = _resolve(sym, "SimulationState", "asset_dist.state", "asset_dist", "state")

    sp = _species(Species, name="A", max_age=200, radius=0.05, reproduction_rate=5, reproduction_radius=2.0)

    size = (20.0, 20.0)
    center = (10.0, 10.0)

    # Start away from edges so reproduction_radius can't leave bounds.
    st = SimulationState([Plant(center, sp, age=1)], size=size, div=10)

    random.seed(0)  # deterministic: first randint(0,5) -> 3 seeds
    st.run_state(1)

    plants = list(st)
    assert len(plants) > 1  # should have spawned at least one

    ages = [p.age for p in plants]
    assert max(ages) == 2
    assert 0 in ages

    for p in plants:
        assert 0.0 <= p.coords[0] <= size[0]
        assert 0.0 <= p.coords[1] <= size[1]


@pytest.mark.integration
def test_run_state_respects_max_population_cap(sym):
    """
    Integration: max_population should prevent runaway spawning.
    We don't assert exact final count (random), but it must not exceed the cap.
    """
    Species = _resolve(sym, "Species", "asset_dist.definitions", "asset_dist", "definitions")
    Plant = _resolve(sym, "Plant", "asset_dist.definitions", "asset_dist", "definitions")
    SimulationState = _resolve(sym, "SimulationState", "asset_dist.state", "asset_dist", "state")

    sp = _species(
        Species,
        name="Cap",
        max_age=200,
        radius=0.03,            
        reproduction_rate=8,
        reproduction_radius=2.0,
        species_density=0.1,
    )

    size = (30.0, 30.0)
    st = SimulationState([Plant((15.0, 15.0), sp, age=0)], size=size, div=10)

    cap = 10
    random.seed(0)
    st.run_state(20, max_population=cap)

    assert len(st) <= cap
    assert len(st) >= 2  # should grow at least a bit with these params


@pytest.mark.integration
def test_run_state_removes_plants_older_than_max_age(sym):
    """
    Integration: plants with age > max_age at the start of a year are removed.
    """
    Species = _resolve(sym, "Species", "asset_dist.definitions", "asset_dist", "definitions")
    Plant = _resolve(sym, "Plant", "asset_dist.definitions", "asset_dist", "definitions")
    SimulationState = _resolve(sym, "SimulationState", "asset_dist.state", "asset_dist", "state")

    sp = _species(Species, name="Old", max_age=1, reproduction_rate=0, reproduction_radius=0.0)

    size = (10.0, 10.0)
    st = SimulationState([Plant((5.0, 5.0), sp, age=2)], size=size, div=5)

    random.seed(0)
    st.run_state(1)

    assert len(st) == 0

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


@pytest.fixture
def defs(sym):
    """Resolve core definitions once."""
    Species = _resolve(sym, "Species", "asset_dist.definitions", "definitions", "asset_dist")
    Plant = _resolve(sym, "Plant", "asset_dist.definitions", "definitions", "asset_dist")
    SimulationState = _resolve(sym, "SimulationState", "asset_dist.state", "state", "asset_dist")
    return Species, Plant, SimulationState


def _species(Species, name="S", *, max_age=10, radius=1.0, reproduction_rate=0, viability_map=None):
    if viability_map is None:
        viability_map = lambda x, y: 1.0  # noqa: E731
    # Keep vt() predictable by removing juvenile penalty in tests.
    return Species(
        name=name,
        max_age=max_age,
        radius=radius,
        reproduction_rate=reproduction_rate,
        juvenile_mortality_depth=0.0,
        juvenile_recovery_age=0.2,
        viability_map=viability_map,
    )


@pytest.mark.unit
def test_state_len_iter_add_remove_consistent(defs):
    Species, Plant, SimulationState = defs

    sp = _species(Species, "A", radius=0.5)
    plants = [Plant((1.0, 1.0), sp, 0), Plant((2.0, 2.0), sp, 0)]
    st = SimulationState(plants, size=(10.0, 10.0), div=10)

    assert len(st) == 2
    assert len(list(st)) == 2

    # remove one, add one
    st.remove(plants[0])
    assert len(st) == 1

    st.add(Plant((3.0, 3.0), sp, 0))
    assert len(st) == 2


@pytest.mark.unit
def test_get_cell_partitions_space_and_handles_edges(defs):
    Species, Plant, SimulationState = defs

    sp = _species(Species, "A")
    st = SimulationState([], size=(10.0, 10.0), div=10)  # cell_width = 1.0

    assert st.get_cell((0.0, 0.0)) == (0, 0)
    assert st.get_cell((0.99, 0.99)) == (0, 0)
    assert st.get_cell((1.0, 1.0)) == (1, 1)

    # Edge exactly at max size should still map inside allocated grid (+1 sizing).
    assert st.get_cell((10.0, 10.0)) == (10, 10)


@pytest.mark.unit
def test_get_nearby_requires_radius_for_coords(defs):
    Species, Plant, SimulationState = defs

    st = SimulationState([], size=(10.0, 10.0), div=10)

    with pytest.raises(TypeError):
        # coords path requires explicit radius
        list(st.get_nearby((1.0, 1.0)))


@pytest.mark.unit
def test_get_nearby_with_plant_uses_species_radius(defs):
    Species, Plant, SimulationState = defs

    sp = _species(Species, "A", radius=1.0)
    a = Plant((1.0, 1.0), sp, 0)
    b = Plant((1.2, 1.0), sp, 0)
    far = Plant((9.0, 9.0), sp, 0)

    st = SimulationState([a, b, far], size=(10.0, 10.0), div=10)

    nearby = list(st.get_nearby(a))  # radius inferred from a.species.radius
    assert b in nearby
    assert far not in nearby


# --------- _evaluate_seed() competition rule ---------

@pytest.mark.unit
def test_evaluate_seed_accepts_and_marks_removal_when_candidate_has_higher_viability(defs):
    Species, Plant, SimulationState = defs

    # Viability depends on x: slightly to the right is much more viable.
    vmap = lambda x, y: 0.1 if x < 0.55 else 0.9  # noqa: E731
    sp = _species(Species, "A", radius=1.0, viability_map=vmap)

    existing = Plant((0.50, 0.50), sp, 20)
    candidate = Plant((0.60, 0.50), sp, 20)  # overlapping (distance 0.1 < 1.0)

    st = SimulationState([existing], size=(10.0, 10.0), div=10)

    pop = {sp: 1}
    total = 1

    ok, removable = st._evaluate_seed(candidate, pop, total)

    assert ok is True
    assert removable == [existing]


@pytest.mark.unit
def test_evaluate_seed_rejects_when_neighbor_has_higher_viability(defs):
    Species, Plant, SimulationState = defs

    vmap = lambda x, y: 0.9 if x < 0.55 else 0.1  # noqa: E731
    sp = _species(Species, "A", radius=1.0, viability_map=vmap)

    existing = Plant((0.50, 0.50), sp, 20)
    candidate = Plant((0.60, 0.50), sp, 20)

    st = SimulationState([existing], size=(10.0, 10.0), div=10)

    pop = {sp: 1}
    total = 1

    ok, removable = st._evaluate_seed(candidate, pop, total)

    assert ok is False
    assert removable == []


@pytest.mark.unit
def test_evaluate_seed_no_neighbors_is_trivially_viable(defs):
    Species, Plant, SimulationState = defs

    sp = _species(Species, "A", radius=1.0)
    candidate = Plant((5.0, 5.0), sp, 0)
    st = SimulationState([], size=(10.0, 10.0), div=10)

    ok, removable = st._evaluate_seed(candidate, {sp: 0}, 1)
    assert ok is True
    assert removable == []


# --------- run_state() semantics ---------

@pytest.mark.unit
def test_run_state_ages_plants_and_removes_only_after_exceeding_max_age(defs):
    Species, Plant, SimulationState = defs

    sp = _species(Species, "A", max_age=2, reproduction_rate=0)
    p = Plant((1.0, 1.0), sp, age=2)  # exactly max_age

    st = SimulationState([p], size=(10.0, 10.0), div=10)

    st.run_state(1)
    # Not removed yet; code removes only if age > max_age before increment.
    assert len(st) == 1
    assert next(iter(st)).age == 3

    st.run_state(1)
    # Now removed because age (3) > max_age (2) at start of year.
    assert len(st) == 0


@pytest.mark.unit
def test_run_state_respects_max_population_cap(defs, monkeypatch):
    Species, Plant, SimulationState = defs

    sp = _species(Species, "A", max_age=100, reproduction_rate=999, radius=0.1)
    p = Plant((1.0, 1.0), sp, age=1)

    st = SimulationState([p], size=(10.0, 10.0), div=10)

    # Deterministic "seed": always tries to add two plants within bounds.
    def seeded(self):
        return [
            Plant((2.0, 2.0), self.species, 0),
            Plant((3.0, 3.0), self.species, 0),
        ]

    monkeypatch.setattr(Plant, "seed", seeded, raising=True)

    st.run_state(5, max_population=2)

    # Start=1 plant. In year 1 it may add at most one new plant before cap triggers.
    assert len(st) == 2


@pytest.mark.unit
def test_run_state_rejects_out_of_bounds_seeds(defs, monkeypatch):
    Species, Plant, SimulationState = defs

    sp = _species(Species, "A", max_age=100, reproduction_rate=999, radius=0.1)
    p = Plant((1.0, 1.0), sp, age=1)

    st = SimulationState([p], size=(10.0, 10.0), div=10)

    def seeded(self):
        # One out-of-bounds, one valid.
        return [
            Plant((-1.0, 2.0), self.species, 0),
            Plant((2.0, 2.0), self.species, 0),
        ]

    monkeypatch.setattr(Plant, "seed", seeded, raising=True)

    st.run_state(1)
    assert len(st) == 2
    coords = {plant.coords for plant in st}
    assert (-1.0, 2.0) not in coords
    assert (2.0, 2.0) in coords

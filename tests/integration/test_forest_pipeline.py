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


def _species(Species, *, name="Canopy", max_age=50, radius=0.8, species_density=0.002):
    # Keep simulation deterministic-ish
    return Species(
        name=name,
        max_age=max_age,
        radius=radius,
        species_density=species_density,
        reproduction_rate=0,         
        reproduction_radius=2.0,
        juvenile_mortality_depth=0.0,
        juvenile_recovery_age=0.2,
        viability_map=lambda x, y: 1.0,  # noqa: E731
    )


@pytest.mark.integration
def test_forest_builder_generate_years_0_produces_initial_state(sym):
    """
    ForestBuilder -> ForestGenerator.generate(years=0):
    - returns a SimulationState
    - plants are within bounds
    - ages are all 0 (no yearly stepping)
    """
    pytest.importorskip("scipy")  # uses scipy.stats.qmc.PoissonDisk

    ForestBuilder = _resolve(sym, "ForestBuilder", "forest.forest_builder", "forest")
    ForestConfig = _resolve(sym, "ForestConfig", "forest.forest_config", "forest")
    SimulationState = _resolve(sym, "SimulationState", "asset_dist.state", "asset_dist", "state")
    Species = _resolve(sym, "Species", "asset_dist.definitions", "asset_dist", "definitions")

    size = (40.0, 40.0)
    sp = _species(Species)

    forest = (
        ForestBuilder()
        .with_size(size)
        .add_species("canopy", sp)
        .build()
    )

    state = forest.generate(ForestConfig(scene_density=1.0, years=0))
    assert isinstance(state, SimulationState)

    plants = list(state)
    assert len(plants) > 0

    for p in plants:
        assert 0.0 <= p.coords[0] <= size[0]
        assert 0.0 <= p.coords[1] <= size[1]
        assert p.age == 0


@pytest.mark.integration
def test_forest_builder_generate_years_1_advances_ages(sym):
    """
    ForestBuilder -> ForestGenerator.generate(years=1):
    - runs the simulation for one year
    - with reproduction_rate=0, the only change should be aging
    """
    pytest.importorskip("scipy")

    ForestBuilder = _resolve(sym, "ForestBuilder", "forest.forest_builder", "forest")
    ForestConfig = _resolve(sym, "ForestConfig", "forest.forest_config", "forest")
    SimulationState = _resolve(sym, "SimulationState", "asset_dist.state", "asset_dist", "state")
    Species = _resolve(sym, "Species", "asset_dist.definitions", "asset_dist", "definitions")

    size = (40.0, 40.0)
    sp = _species(Species, max_age=10)

    forest = (
        ForestBuilder()
        .with_size(size)
        .add_species("canopy", sp)
        .build()
    )

    state = forest.generate(ForestConfig(scene_density=1.0, years=1))
    assert isinstance(state, SimulationState)

    plants = list(state)
    assert len(plants) > 0

    # With reproduction_rate=0, everyone should be age 1 after one year.
    ages = {p.age for p in plants}
    assert ages == {1}

    for p in plants:
        assert 0.0 <= p.coords[0] <= size[0]
        assert 0.0 <= p.coords[1] <= size[1]


@pytest.mark.integration
def test_forest_builder_generate_years_respects_bounds_even_with_small_size(sym):
    """
    Regression guard  small scenes + PoissonDisk 
    """
    pytest.importorskip("scipy")

    ForestBuilder = _resolve(sym, "ForestBuilder", "forest.forest_builder", "forest")
    ForestConfig = _resolve(sym, "ForestConfig", "forest.forest_config", "forest")
    Species = _resolve(sym, "Species", "asset_dist.definitions", "asset_dist", "definitions")

    size = (12.0, 12.0)
    sp = _species(Species, radius=0.6, species_density=0.01)  

    forest = (
        ForestBuilder()
        .with_size(size)
        .add_species("canopy", sp)
        .build()
    )

    state = forest.generate(ForestConfig(scene_density=1.0, years=1))
    for p in state:
        assert 0.0 <= p.coords[0] <= size[0]
        assert 0.0 <= p.coords[1] <= size[1]

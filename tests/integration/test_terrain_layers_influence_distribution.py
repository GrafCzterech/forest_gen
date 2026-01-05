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


class _TerrainStub:
    """Minimal Terrain-like object for ForestGenerator wiring tests."""
    def __init__(self, *, size: float, resolution: float, moisture=None, slope=None, aspect=None):
        self.moisture = moisture
        self.slope = slope
        self.aspect = aspect

        class _Cfg:
            def __init__(self, size, resolution):
                self.size = size
                self.resolution = resolution

        self.config = _Cfg(size, resolution)


@pytest.mark.integration
def test_viability_layers_are_applied_to_species_viability_map(sym):
    """
    Integration: ForestGenerator should apply TerrainViabilityMap layers by wrapping
    Species.viability_map via DistributionBuilder.

    This test does NOT rely on distribution outcomes (which currently won't bias).
    It proves the wiring and sampling math are correct end-to-end.
    """
    pytest.importorskip("scipy")  # Simulation uses scipy.stats.qmc.PoissonDisk

    ForestGenerator = _resolve(sym, "ForestGenerator", "forest.forest_generator", "forest")
    Species = _resolve(sym, "Species", "asset_dist.definitions", "asset_dist", "definitions")

    size = 40.0
    res = 1.0
    n = int(round(size / res)) + 1  # matches TerrainConfig.rows/cols convention

    # A sharp mask: left half low viability, right half high viability.
    mask = np.ones((n, n), dtype=np.float32)
    mid = n // 2
    mask[:, :mid] = 0.05
    mask[:, mid:] = 1.0

    # Built-in "moisture" layer in ForestGenerator will be included if not None.
    # Set it to ones so it doesn't change the mask.
    moisture = np.ones((n, n), dtype=np.float32)
    terrain = _TerrainStub(size=size, resolution=res, moisture=moisture)

    sp = Species(
        name="Test",
        max_age=50,
        radius=1.0,
        species_density=0.01,
        reproduction_rate=0,
        juvenile_mortality_depth=0.0,
        juvenile_recovery_age=0.2,
        viability_map=lambda x, y: 1.0,  # noqa: E731
    )

    # Constructing ForestGenerator triggers DistributionBuilder.build(),
    # which wraps sp.viability_map in-place.
    _ = ForestGenerator(
        size=(size, size),
        species={"canopy": {sp}},
        terrain=terrain,
        terrain_layers={"mask": mask},
        layer_combiner=None,  # default: multiply layers
    )

    left = sp.viability_map(5.0, 5.0)    # column ~ 5 => low region
    right = sp.viability_map(30.0, 5.0)  # column ~ 30 => high region

    assert 0.0 <= left <= 1.0
    assert 0.0 <= right <= 1.0
    assert left < right
    assert left == pytest.approx(0.05, abs=1e-6)
    assert right == pytest.approx(1.0, abs=1e-6)


@pytest.mark.integration
def test_viability_layers_bias_distribution_over_time(sym):
    """
    Intended integration (future): terrain viability should bias where plants persist.

    This is xfail for now because of the current mechanics described in the reason.
    """
    pytest.importorskip("scipy")

    ForestGenerator = _resolve(sym, "ForestGenerator", "forest.forest_generator", "forest")
    ForestConfig = _resolve(sym, "ForestConfig", "forest.forest_config", "forest")
    Species = _resolve(sym, "Species", "asset_dist.definitions", "asset_dist", "definitions")

    size = 60.0
    res = 1.0
    n = int(round(size / res)) + 1
    mid_x = size / 2.0

    mask = np.ones((n, n), dtype=np.float32)
    mid = n // 2
    mask[:, :mid] = 0.05
    mask[:, mid:] = 1.0

    terrain = _TerrainStub(size=size, resolution=res, moisture=np.ones((n, n), dtype=np.float32))

    sp = Species(
        name="Test",
        max_age=25,
        radius=1.0,
        species_density=0.02,
        reproduction_rate=5,
        reproduction_radius=3.0,
        juvenile_mortality_depth=0.0,
        juvenile_recovery_age=0.2,
        viability_map=lambda x, y: 1.0,  # noqa: E731
    )

    gen = ForestGenerator(
        size=(size, size),
        species={"canopy": {sp}},
        terrain=terrain,
        terrain_layers={"mask": mask},
        layer_combiner=None,
    )

    state = gen.generate(ForestConfig(scene_density=1.0, years=3))
    xs = [p.coords[0] for p in state]
    assert max(xs) > mid_x, f"max x is {max(xs):.3f} (looks like unit-cube sampling)"
    left = sum(1 for p in state if p.coords[0] < mid_x)
    right = sum(1 for p in state if p.coords[0] >= mid_x)

    # Intended expectation (will currently fail): more plants in the high-viability half.
    assert right > left * 1.5

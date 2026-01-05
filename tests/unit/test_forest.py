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


def _resolve_module(PKG: str, rel: str):
    import importlib
    return importlib.import_module(f"{PKG}.{rel}")


class _BuilderStub:
    """Captures calls made by ForestGenerator to DistributionBuilder."""

    def __init__(self):
        self.size = None
        self.added = []  # list[(kind, species)]
        self.terrain_layers_call = None  # (layers, resolution, combine)
        self.built = False
        self._generator = _GeneratorStub()

    def with_size(self, size):
        self.size = size
        return self

    def add_species(self, kind, sp):
        self.added.append((kind, sp))
        return self

    def with_terrain_viability_layers(self, layers, resolution, combine=None):
        self.terrain_layers_call = (layers, resolution, combine)
        return self

    def build(self):
        self.built = True
        return self._generator


class _GeneratorStub:
    """Acts as DistributionGenerator; returns sentinel and captures config passed."""
    def __init__(self):
        self.last_config = None
        self.calls = 0
        self.sentinel = object()

    def generate(self, config):
        self.calls += 1
        self.last_config = config
        return self.sentinel


class _TerrainStub:
    def __init__(self, *, resolution=1.0, moisture=None, slope=None, aspect=None, size=100.0):
        self.moisture = moisture
        self.slope = slope
        self.aspect = aspect

        class _Cfg:
            def __init__(self, resolution, size):
                self.resolution = resolution
                self.size = size

        self.config = _Cfg(resolution, size)


@pytest.mark.unit
def test_forest_generator_builds_distribution_builder_and_forwards_species(sym, monkeypatch):
    ForestGenerator = _resolve(sym, "ForestGenerator", "forest.forest_generator", "forest")
    Species = _resolve(sym, "Species", "asset_dist.definitions", "asset_dist", "definitions")

    # Patch DistributionBuilder in the ForestGenerator module.
    fg_mod = __import__(ForestGenerator.__module__, fromlist=["_"])
    builder = _BuilderStub()
    monkeypatch.setattr(fg_mod, "DistributionBuilder", lambda: builder)

    # Minimal species
    sp_a = Species("A", max_age=10)
    sp_b = Species("B", max_age=10)
    species = {"canopy": {sp_a}, "understory": {sp_b}}

    fg = ForestGenerator(size=(12.0, 34.0), species=species, terrain=None)

    assert fg is not None
    assert builder.size == (12.0, 34.0)

    # All species should be forwarded into builder.add_species
    assert set(builder.added) == {("canopy", sp_a), ("understory", sp_b)}
    assert builder.built is True

    # No terrain => no terrain viability layers call
    assert builder.terrain_layers_call is None


@pytest.mark.unit
def test_forest_generator_applies_filtered_terrain_layers_and_combiner(sym, monkeypatch):
    ForestGenerator = _resolve(sym, "ForestGenerator", "forest.forest_generator", "forest")
    Species = _resolve(sym, "Species", "asset_dist.definitions", "asset_dist", "definitions")

    fg_mod = __import__(ForestGenerator.__module__, fromlist=["_"])
    builder = _BuilderStub()
    monkeypatch.setattr(fg_mod, "DistributionBuilder", lambda: builder)

    sp = Species("A", max_age=10)
    species = {"canopy": {sp}}

    moisture = np.ones((4, 4), dtype=np.float32)
    aspect = np.zeros((4, 4), dtype=np.float32)
    terrain = _TerrainStub(resolution=2.5, moisture=moisture, slope=None, aspect=aspect)

    def comb(values):
        return 0.123

    ForestGenerator(
        size=(50.0, 50.0),
        species=species,
        terrain=terrain,
        terrain_layers=None,
        layer_combiner=comb,
    )

    assert builder.terrain_layers_call is not None
    layers, resolution, combine = builder.terrain_layers_call

    # Only non-None layers should be forwarded
    assert set(layers.keys()) == {"moisture"}
    assert layers["moisture"] is moisture


    # Resolution and combiner must be forwarded exactly
    assert resolution == pytest.approx(2.5)
    assert combine is comb


@pytest.mark.unit
def test_forest_generator_merges_and_overrides_terrain_layers(sym, monkeypatch):
    ForestGenerator = _resolve(sym, "ForestGenerator", "forest.forest_generator", "forest")
    Species = _resolve(sym, "Species", "asset_dist.definitions", "asset_dist", "definitions")

    fg_mod = __import__(ForestGenerator.__module__, fromlist=["_"])
    builder = _BuilderStub()
    monkeypatch.setattr(fg_mod, "DistributionBuilder", lambda: builder)

    sp = Species("A", max_age=10)
    species = {"canopy": {sp}}

    moisture_builtin = np.full((2, 2), 0.1, dtype=np.float32)
    moisture_override = np.full((2, 2), 0.9, dtype=np.float32)
    custom = np.full((2, 2), 0.7, dtype=np.float32)

    terrain = _TerrainStub(resolution=1.0, moisture=moisture_builtin, slope=None, aspect=None)
    terrain_layers = {
        "moisture": moisture_override,  # should override builtin moisture
        "custom": custom,
        "ignored": None,                # should be filtered out
    }

    ForestGenerator(
        size=(10.0, 10.0),
        species=species,
        terrain=terrain,
        terrain_layers=terrain_layers,
        layer_combiner=None,
    )

    layers, resolution, combine = builder.terrain_layers_call
    assert resolution == pytest.approx(1.0)
    assert combine is not None or combine is None  # don’t care; just explicit

    assert set(layers.keys()) == {"moisture", "custom"}
    assert layers["moisture"] is moisture_override
    assert layers["custom"] is custom
    assert "ignored" not in layers


@pytest.mark.unit
def test_forest_generator_generate_forwards_forest_config(sym, monkeypatch):
    ForestGenerator = _resolve(sym, "ForestGenerator", "forest.forest_generator", "forest")
    ForestConfig = _resolve(sym, "ForestConfig", "forest.forest_config", "forest")

    DistributionConfig = _resolve(
        sym,
        "DistributionConfig",
        "asset_dist.distribution_config",
        "asset_dist",
        "distribution_config",
    )

    fg_mod = __import__(ForestGenerator.__module__, fromlist=["_"])
    builder = _BuilderStub()
    monkeypatch.setattr(fg_mod, "DistributionBuilder", lambda: builder)

    # No need for real Species here; ForestGenerator only iterates and forwards.
    # Use empty species dict to keep it minimal and still validate generate() forwarding.
    fg = ForestGenerator(size=(5.0, 5.0), species={}, terrain=None)

    cfg = ForestConfig(scene_density=2.5, years=7)
    out = fg.generate(cfg)

    assert out is builder._generator.sentinel
    assert builder._generator.calls == 1
    assert isinstance(builder._generator.last_config, DistributionConfig)
    assert builder._generator.last_config.scene_density == pytest.approx(2.5)
    assert builder._generator.last_config.years == 7


@pytest.mark.unit
def test_forest_builder_passes_state_into_forest_generator(sym, monkeypatch):
    ForestBuilder = _resolve(sym, "ForestBuilder", "forest.forest_builder", "forest")
    Species = _resolve(sym, "Species", "asset_dist.definitions", "asset_dist", "definitions")
    ForestGenerator = _resolve(sym, "ForestGenerator", "forest.forest_generator", "forest")

    # Patch ForestGenerator in the builder module to capture constructor args
    fb_mod = __import__(ForestBuilder.__module__, fromlist=["_"])
    captured = {}

    class _FGSpy:
        def __init__(self, size, species, terrain, terrain_layers, layer_combiner):
            captured["size"] = size
            captured["species"] = species
            captured["terrain"] = terrain
            captured["terrain_layers"] = terrain_layers
            captured["layer_combiner"] = layer_combiner

    monkeypatch.setattr(fb_mod, "ForestGenerator", _FGSpy)

    sp = Species("A", max_age=10)
    terrain = _TerrainStub(resolution=1.0)
    layers = {"x": np.ones((2, 2), dtype=np.float32)}
    comb = lambda d: 1.0  # noqa: E731

    b = (
        ForestBuilder()
        .with_size((9.0, 8.0))
        .with_terrain(terrain)
        .with_terrain_viability_layers(layers, combine=comb)
        .add_species("canopy", sp)
    )

    b.build()

    assert captured["size"] == (9.0, 8.0)
    assert captured["terrain"] is terrain
    assert captured["terrain_layers"] is layers
    assert captured["layer_combiner"] is comb
    assert "canopy" in captured["species"]
    assert sp in captured["species"]["canopy"]

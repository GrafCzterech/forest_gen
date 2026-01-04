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


def _resolve_grass_symbols(sym):
    GrassDistributor = _resolve(
        sym,
        "GrassDistributor",
        "asset_dist.grass",
        "asset_dist.grass_distributor",
        "grass",
    )
    TreeProximityMap = _resolve(
        sym,
        "TreeProximityMap",
        "asset_dist.grass",
        "asset_dist.grass_distributor",
        "grass",
    )
    PatchyGrassMap = _resolve(
        sym,
        "PatchyGrassMap",
        "asset_dist.grass",
        "asset_dist.grass_distributor",
        "grass",
    )
    return GrassDistributor, TreeProximityMap, PatchyGrassMap


class _TerrainStub:
    def __init__(self, *, size=100.0, resolution=1.0, moisture=None, slope=None, aspect=None):
        self.moisture = moisture
        self.slope = slope
        self.aspect = aspect

        class _Cfg:
            def __init__(self, size, resolution):
                self.size = size
                self.resolution = resolution

        self.config = _Cfg(size, resolution)


# -------------------- TreeProximityMap --------------------

@pytest.mark.unit
def test_tree_proximity_map_no_trees_is_one(sym):
    pytest.importorskip("opensimplex")
    pytest.importorskip("scipy")

    _, TreeProximityMap, _ = _resolve_grass_symbols(sym)

    m = TreeProximityMap([])
    assert m(0.0, 0.0) == pytest.approx(1.0)
    assert m(123.0, -5.0) == pytest.approx(1.0)


@pytest.mark.unit
def test_tree_proximity_map_hard_radius_is_zero(sym):
    pytest.importorskip("opensimplex")
    pytest.importorskip("scipy")

    _, TreeProximityMap, _ = _resolve_grass_symbols(sym)

    m = TreeProximityMap(tree_positions=[(0.0, 0.0)], hard_radius=2.0, falloff_radius=6.0)
    assert m(0.0, 0.0) == pytest.approx(0.0)
    assert m(1.99, 0.0) == pytest.approx(0.0)


@pytest.mark.unit
def test_tree_proximity_map_beyond_falloff_is_one(sym):
    pytest.importorskip("opensimplex")
    pytest.importorskip("scipy")

    _, TreeProximityMap, _ = _resolve_grass_symbols(sym)

    m = TreeProximityMap(tree_positions=[(0.0, 0.0)], hard_radius=2.0, falloff_radius=6.0)
    assert m(6.0, 0.0) == pytest.approx(1.0)
    assert m(100.0, 0.0) == pytest.approx(1.0)


@pytest.mark.unit
def test_tree_proximity_map_linear_falloff_between_radii(sym):
    pytest.importorskip("opensimplex")
    pytest.importorskip("scipy")

    _, TreeProximityMap, _ = _resolve_grass_symbols(sym)

    m = TreeProximityMap(tree_positions=[(0.0, 0.0)], hard_radius=2.0, falloff_radius=6.0)

    assert m(4.0, 0.0) == pytest.approx(0.5, abs=1e-12)


@pytest.mark.unit
def test_tree_proximity_map_uses_closest_tree(sym):
    pytest.importorskip("opensimplex")
    pytest.importorskip("scipy")

    _, TreeProximityMap, _ = _resolve_grass_symbols(sym)

    m = TreeProximityMap(tree_positions=[(0.0, 0.0), (100.0, 0.0)], hard_radius=2.0, falloff_radius=6.0)

    # closest tree  at origin
    assert m(4.0, 0.0) == pytest.approx(0.5, abs=1e-12)

    # closest tree  at x=100
    assert m(96.0, 0.0) == pytest.approx(0.5, abs=1e-12)


# -------------------- PatchyGrassMap --------------------

@pytest.mark.unit
def test_patchy_grass_map_thresholding(sym, monkeypatch):
    pytest.importorskip("opensimplex")
    pytest.importorskip("scipy")

    _, _, PatchyGrassMap = _resolve_grass_symbols(sym)
    mod = __import__(PatchyGrassMap.__module__, fromlist=["_"])

    # Stub OpenSimplex  control noise2 
    class _Noise:
        def __init__(self, seed):  # noqa: ARG002
            self.value = 0.0

        def noise2(self, x, y):  # noqa: ARG002
            return self.value

    class _OpenSimplexStub:
        def __init__(self, seed):  # noqa: ARG002
            self._n = _Noise(seed)

        def noise2(self, x, y):
            return self._n.noise2(x, y)

    monkeypatch.setattr(mod, "OpenSimplex", _OpenSimplexStub)

    pm = PatchyGrassMap(scale=0.2, seed=123)

    # noise > 0 => 1.0
    pm.noise._n.value = 0.001
    assert pm(1.0, 2.0) == 1.0

    # noise == 0 => 0.0 (strict > 0 check)
    pm.noise._n.value = 0.0
    assert pm(1.0, 2.0) == 0.0

    # noise < 0 => 0.0
    pm.noise._n.value = -0.5
    assert pm(1.0, 2.0) == 0.0


# -------------------- GrassDistributor internals --------------------

@pytest.mark.unit
def test_grass_distributor_terrain_layers_include_moisture_and_slope_viability(sym):
    pytest.importorskip("opensimplex")
    pytest.importorskip("scipy")

    GrassDistributor, _, _ = _resolve_grass_symbols(sym)

    moisture = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    slope = np.array([[0.0, 10.0], [20.0, 40.0]], dtype=np.float32)  

    terrain = _TerrainStub(size=50.0, resolution=2.0, moisture=moisture, slope=slope)

    d = GrassDistributor(terrain, tree_positions=[])
    layers = d._terrain_layers()

    assert "moisture" in layers
    assert layers["moisture"] is moisture

    assert "slope_viability" in layers
    expected = 1.0 - np.clip(slope / float(np.max(slope)), 0.0, 1.0)
    assert np.allclose(layers["slope_viability"], expected)


@pytest.mark.unit
def test_grass_distributor_terrain_layers_omits_slope_viability_when_max_slope_zero(sym):
    pytest.importorskip("opensimplex")
    pytest.importorskip("scipy")

    GrassDistributor, _, _ = _resolve_grass_symbols(sym)

    slope = np.zeros((3, 3), dtype=np.float32)
    terrain = _TerrainStub(size=10.0, resolution=1.0, moisture=None, slope=slope)

    d = GrassDistributor(terrain, tree_positions=[])
    layers = d._terrain_layers()

    assert "moisture" not in layers
    assert "slope_viability" not in layers 


@pytest.mark.unit
def test_grass_distributor_combine_layers_is_multiplicative(sym):
    pytest.importorskip("opensimplex")
    pytest.importorskip("scipy")

    GrassDistributor, _, _ = _resolve_grass_symbols(sym)

    terrain = _TerrainStub()
    d = GrassDistributor(terrain, tree_positions=[])

    assert d._combine_layers({"a": 0.5, "b": 0.2, "c": 1.0}) == pytest.approx(0.1)


@pytest.mark.unit
def test_grass_species_viability_is_product_of_patchiness_and_tree_map(sym):
    pytest.importorskip("opensimplex")
    pytest.importorskip("scipy")

    GrassDistributor, _, _ = _resolve_grass_symbols(sym)
    Species = _resolve(sym, "Species", "asset_dist.definitions", "asset_dist", "definitions")

    terrain = _TerrainStub(size=10.0, resolution=1.0)
    d = GrassDistributor(terrain, tree_positions=[])

    d.patchiness = lambda x, y: 0.2  # noqa: E731
    d.tree_map = lambda x, y: 0.5  # noqa: E731

    sp = d._grass_species()
    assert isinstance(sp, Species)
    assert sp.name == "Grass"
    assert sp.viability_map(1.0, 1.0) == pytest.approx(0.1)


# -------------------- GrassDistributor.generate wiring --------------------

@pytest.mark.unit
def test_grass_distributor_generate_wires_forest_builder_correctly(sym, monkeypatch):
    pytest.importorskip("opensimplex")
    pytest.importorskip("scipy")

    GrassDistributor, _, _ = _resolve_grass_symbols(sym)
    ForestConfig = _resolve(sym, "ForestConfig", "forest.forest_config", "forest")

    mod = __import__(GrassDistributor.__module__, fromlist=["_"])

    captured = {
        "size": None,
        "terrain": None,
        "layers": None,
        "combine": None,
        "species_kind": None,
        "species_obj": None,
        "forest_generate_called": False,
        "forest_config": None,
    }

    class _ForestStub:
        def generate(self, config):
            captured["forest_generate_called"] = True
            captured["forest_config"] = config
            return object()

    class _ForestBuilderSpy:
        def with_size(self, size):
            captured["size"] = size
            return self

        def with_terrain(self, terrain):
            captured["terrain"] = terrain
            return self

        def with_terrain_viability_layers(self, layers, combine=None):
            captured["layers"] = layers
            captured["combine"] = combine
            return self

        def add_species(self, kind, species):
            captured["species_kind"] = kind
            captured["species_obj"] = species
            return self

        def build(self):
            return _ForestStub()

    monkeypatch.setattr(mod, "ForestBuilder", _ForestBuilderSpy)

    moisture = np.ones((4, 4), dtype=np.float32)
    slope = np.ones((4, 4), dtype=np.float32)
    terrain = _TerrainStub(size=33.0, resolution=1.0, moisture=moisture, slope=slope)

    d = GrassDistributor(
        terrain,
        tree_positions=[(1.0, 1.0)],
        patch_scale=0.12,
        hard_radius=2.0,
        falloff_radius=6.0,
    )
    cfg = ForestConfig(scene_density=1.2, years=3)

    out = d.generate(cfg)

    assert out is not None
    assert captured["size"] == (terrain.config.size, terrain.config.size)
    assert captured["terrain"] is terrain

    assert isinstance(captured["layers"], dict)
    assert "moisture" in captured["layers"]
    assert "slope_viability" in captured["layers"]

    assert callable(captured["combine"])
    assert captured["species_kind"] == "grass"
    assert captured["species_obj"].name == "Grass"

    assert captured["forest_generate_called"] is True
    assert captured["forest_config"] is cfg

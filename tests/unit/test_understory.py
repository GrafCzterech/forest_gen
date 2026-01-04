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


def _resolve_understory_symbols(sym):
    # Adjust/add paths here if your package layout differs.
    UnderstoryDistributor = _resolve(
        sym,
        "UnderstoryDistributor",
        "asset_dist.understory",
        "asset_dist.understory_distributor",
        "understory",
    )
    CanopyShadeMap = _resolve(
        sym,
        "CanopyShadeMap",
        "asset_dist.understory",
        "asset_dist.understory_distributor",
        "understory",
    )
    PatchyUnderstoryMap = _resolve(
        sym,
        "PatchyUnderstoryMap",
        "asset_dist.understory",
        "asset_dist.understory_distributor",
        "understory",
    )
    return UnderstoryDistributor, CanopyShadeMap, PatchyUnderstoryMap


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


# -------------------- CanopyShadeMap --------------------

@pytest.mark.unit
def test_canopy_shade_map_no_canopy_is_all_ones(sym):
    _, CanopyShadeMap, _ = _resolve_understory_symbols(sym)

    m = CanopyShadeMap([])
    assert m(0.0, 0.0) == pytest.approx(1.0)
    assert m(123.0, -5.0) == pytest.approx(1.0)


@pytest.mark.unit
def test_canopy_shade_map_avoid_radius_is_hard_zero(sym):
    _, CanopyShadeMap, _ = _resolve_understory_symbols(sym)

    m = CanopyShadeMap(canopy_positions=[(0.0, 0.0)], avoid_radius=1.5, preferred_distance=4.0, falloff_radius=9.0)

    assert m(0.0, 0.0) == pytest.approx(0.0)
    assert m(1.0, 0.0) == pytest.approx(0.0)  # inside avoid radius


@pytest.mark.unit
def test_canopy_shade_map_peaks_at_preferred_distance(sym):
    _, CanopyShadeMap, _ = _resolve_understory_symbols(sym)

    preferred = 4.0
    m = CanopyShadeMap(canopy_positions=[(0.0, 0.0)], avoid_radius=1.5, preferred_distance=preferred, falloff_radius=9.0)

    # At preferred distance, gaussian = 1 and tail = 1 -> expected ~1
    assert m(preferred, 0.0) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.unit
def test_canopy_shade_map_far_beyond_falloff_goes_to_zero(sym):
    _, CanopyShadeMap, _ = _resolve_understory_symbols(sym)

    # Choose distances so tail clamps to 0 and gaussian is tiny.
    preferred = 4.0
    falloff = 10.0
    m = CanopyShadeMap(canopy_positions=[(0.0, 0.0)], avoid_radius=1.0, preferred_distance=preferred, falloff_radius=falloff)

    # spread = falloff - peak = 6.  pick nearest=falloff+spread -> tail=0
    x = falloff + (falloff - preferred)
    assert m(x, 0.0) == pytest.approx(0.0, abs=1e-6)


# -------------------- PatchyUnderstoryMap --------------------

@pytest.mark.unit
def test_patchy_understory_map_thresholding(sym, monkeypatch):
    pytest.importorskip("opensimplex")

    _, _, PatchyUnderstoryMap = _resolve_understory_symbols(sym)

    # Patch OpenSimplex in the module to control noise2 outputs.
    mod = __import__(PatchyUnderstoryMap.__module__, fromlist=["_"])

    class _Noise:
        def __init__(self, seed):  # noqa: ARG002
            pass

        def noise2(self, x, y):  # noqa: ARG002
            return self.value

    class _OpenSimplexStub:
        def __init__(self, seed):  # noqa: ARG002
            self._noise = _Noise(seed)

        def noise2(self, x, y):
            return self._noise.noise2(x, y)

    # Install stub
    monkeypatch.setattr(mod, "OpenSimplex", _OpenSimplexStub)

    pm = PatchyUnderstoryMap(scale=0.1, threshold=0.35, seed=123)

    # Above threshold -> 1
    pm.noise._noise.value = 0.8
    assert pm(1.0, 2.0) == 1.0

    # Below threshold -> 0
    pm.noise._noise.value = 0.1
    assert pm(1.0, 2.0) == 0.0


# -------------------- UnderstoryDistributor internals --------------------

@pytest.mark.unit
def test_understory_distributor_terrain_layers_include_moisture_and_slope_viability(sym):
    UnderstoryDistributor, _, _ = _resolve_understory_symbols(sym)

    moisture = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    slope = np.array([[0.0, 10.0], [20.0, 40.0]], dtype=np.float32)  # max=40

    terrain = _TerrainStub(size=50.0, resolution=2.0, moisture=moisture, slope=slope)

    d = UnderstoryDistributor(terrain, canopy_positions=[])
    layers = d._terrain_layers()

    assert "moisture" in layers
    assert layers["moisture"] is moisture

    assert "slope_viability" in layers
    expected = 1.0 - np.clip(slope / float(np.max(slope)), 0.0, 1.0)
    assert np.allclose(layers["slope_viability"], expected)


@pytest.mark.unit
def test_understory_distributor_terrain_layers_omits_slope_viability_when_max_slope_zero(sym):
    UnderstoryDistributor, _, _ = _resolve_understory_symbols(sym)

    slope = np.zeros((3, 3), dtype=np.float32)
    terrain = _TerrainStub(size=10.0, resolution=1.0, moisture=None, slope=slope)

    d = UnderstoryDistributor(terrain, canopy_positions=[])
    layers = d._terrain_layers()

    assert "moisture" not in layers
    assert "slope_viability" not in layers  # avoids divide-by-zero / useless layer


@pytest.mark.unit
def test_understory_species_viability_is_product_of_patchiness_and_canopy(sym):
    UnderstoryDistributor, _, _ = _resolve_understory_symbols(sym)
    Species = _resolve(sym, "Species", "asset_dist.definitions", "asset_dist", "definitions")

    terrain = _TerrainStub(size=10.0, resolution=1.0)
    d = UnderstoryDistributor(terrain, canopy_positions=[])

    # Override maps to make viability deterministic
    d.patchiness = lambda x, y: 0.2  # noqa: E731
    d.canopy_map = lambda x, y: 0.5  # noqa: E731

    sp = d._understory_species()
    assert isinstance(sp, Species)
    assert sp.name == "Understory"
    assert sp.viability_map(1.0, 1.0) == pytest.approx(0.1)


# -------------------- UnderstoryDistributor.generate wiring --------------------

@pytest.mark.unit
def test_understory_distributor_generate_wires_forest_builder_correctly(sym, monkeypatch):
    UnderstoryDistributor, _, _ = _resolve_understory_symbols(sym)
    ForestConfig = _resolve(sym, "ForestConfig", "forest.forest_config", "forest")

    mod = __import__(UnderstoryDistributor.__module__, fromlist=["_"])

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

    d = UnderstoryDistributor(terrain, canopy_positions=[(1.0, 1.0)], patch_scale=0.12, patch_threshold=0.45)
    cfg = ForestConfig(scene_density=1.2, years=3)

    out = d.generate(cfg)

    assert out is not None
    assert captured["size"] == (terrain.config.size, terrain.config.size)
    assert captured["terrain"] is terrain

    assert isinstance(captured["layers"], dict)
    assert "moisture" in captured["layers"]
    assert "slope_viability" in captured["layers"]

    assert callable(captured["combine"])
    assert captured["species_kind"] == "understory"
    assert captured["species_obj"].name == "Understory"

    assert captured["forest_generate_called"] is True
    assert captured["forest_config"] is cfg

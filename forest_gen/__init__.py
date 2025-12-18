"""forest_gen – procedural forest-generation toolkit"""

__all__ = [
    "ForestGenSpec",
    "TraversabilityMapBuilder",
    "TraversabilityConfig",
]

def __getattr__(name):
    if name == "ForestGenSpec":
        from .scene import ForestGenSpec

        return ForestGenSpec
    if name == "TraversabilityMapBuilder":
        from .travelsibilitymap import TraversabilityMapBuilder

        return TraversabilityMapBuilder
    if name == "TraversabilityConfig":
        from .travelsibilitymap import TraversabilityConfig

        return TraversabilityConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
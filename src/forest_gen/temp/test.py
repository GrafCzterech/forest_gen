from forest_gen.temp.Terrain.TerrainBuilder import TerrainBuilder
from forest_gen.temp.Terrain.TerrainConfig import TerrainConfig
from forest_gen.temp.Forest.ForestBuilder import ForestBuilder
from forest_gen.temp.Forest.ForestConfig import ForestConfig
from forest_gen.temp.definitions import Species

builder = (TerrainBuilder()
           .with_noise("fractal")
           .with_microrelief(True)
           .with_moisture_model({"flow":0.6,"slope":0.2,"aspect":0.2})
           .with_exporter("glb", resolution=2.0, max_elevation=100.0))

generator = builder.build()
config = TerrainConfig(rows=100, cols=100, resolution=5.0)
generator.generate(config)
# generator.export("output.terrain.glb")
# generator.visualize_all()

# Build a simple forest using the generated terrain moisture
forest_builder = (
    ForestBuilder()
    .with_size((config.cols * config.resolution, config.rows * config.resolution))
    .add_species("trees", Species("oak", 5, 0.02))
    .with_terrain(generator)
)
forest = forest_builder.build()
forest_state = forest.generate(ForestConfig())
print(f"Generated {len(tuple(forest_state))} plants")
from forest_gen.temp.TerrainBuilder import TerrainBuilder
from forest_gen.temp.TerrainConfig import TerrainConfig


builder = (TerrainBuilder()
           .with_noise("fractal")
           .with_microrelief(True)
           .with_moisture_model({"flow":0.6,"slope":0.2,"aspect":0.2})
           .with_exporter("glb", resolution=2.0, max_elevation=100.0))

generator = builder.build()
config = TerrainConfig(rows=100, cols=100, resolution=5.0)
generator.generate(config)
generator.export("output.terrain.glb")
generator.visualize_all()
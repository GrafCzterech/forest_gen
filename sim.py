import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Script that will render a scene defined using forest_gen"
)
parser.add_argument(
    "--asset_path",
    type=str,
    default="./models",
    help="Path to assets",
)
parser.add_argument("--size", type=int, default=50, help="Size of the scene")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

simulation_app = AppLauncher(args_cli).app


from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_assets.robots.spot import SPOT_CFG

from forest_gen import ForestGenSpec

if __name__ == "__main__":
    sim_cfg = SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view((0.0, 0.0, 5.0), (1.0, 1.0, 4.0))

    generator = ForestGenSpec(size=args_cli.size, asset_path=args_cli.asset_path)
    scene_factory = generator.create_instance(num_envs=1, env_spacing=1.0)
    my_scene_cfg = scene_factory.get_scene(SPOT_CFG)

    scene = InteractiveScene(my_scene_cfg)

    sim.reset()

    sim_dt = sim.get_physics_dt()
    sim.pause()

    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
    simulation_app.close()

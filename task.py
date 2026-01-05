from isaaclab.envs import ViewerCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab_assets.robots.spot import SPOT_CFG
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.flat_env_cfg import (
    SpotActionsCfg,
    SpotCommandsCfg,
    SpotEventCfg,
    SpotObservationsCfg,
    SpotRewardsCfg,
    SpotTerminationsCfg,
)
from stripe_kit import TrainingSpec

from forest_gen import ForestGenSpec

task_spec = TrainingSpec(
    scene=ForestGenSpec(asset_path="./models", size=50),
    robot=SPOT_CFG,
    actions=SpotActionsCfg(),
    observations=SpotObservationsCfg(),
    events=SpotEventCfg(),
    rewards=SpotRewardsCfg(),
    terminations=SpotTerminationsCfg(),
    commands=SpotCommandsCfg(),
    sensors={
        "contact_forces": ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/robot/.*",
            history_length=3,
            track_air_time=True,
        )
    },
).to_env_cfg(
    ViewerCfg(
        eye=(3, 3, 3),
        origin_type="asset_root",
        asset_name="robot",
    ),
    4,
    10,
)

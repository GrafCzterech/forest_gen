from isaaclab.envs import ViewerCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab_assets.robots.spot import SPOT_CFG
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.flat_env_cfg import (
    SpotActionsCfg,
    SpotCommandsCfg,
    SpotEventCfg,
    SpotObservationsCfg,
    SpotRewardsCfg,
)
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from stripe_kit import TrainingSpec

from forest_gen import ForestGenSpec

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body", ".*leg"]), "threshold": 1.0},
    )


task_spec = TrainingSpec(
    scene=ForestGenSpec(asset_path="./models", size=50),
    robot=SPOT_CFG,
    actions=SpotActionsCfg(),
    observations=SpotObservationsCfg(),
    events=SpotEventCfg(),
    rewards=SpotRewardsCfg(),
    terminations=TerminationsCfg(),
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

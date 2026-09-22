from isaaclab.assets import ArticulationCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils import configclass
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from carrot_sim.arena_libero import mdp
from carrot_sim.arena_libero.config import ArenaLiberoConfig


def robot_selection() -> SceneEntityCfg:
    return SceneEntityCfg(
        "robot",
        body_names=["panda_hand"],
        joint_names=["panda_finger_joint1", "panda_finger_joint2"],
        preserve_order=True,
    )


@configclass
class PolicyCfg(ObservationGroupCfg):
    state = ObservationTermCfg(func=mdp.tcp_state, params={"asset_cfg": robot_selection()})
    image = ObservationTermCfg(func=mdp.rgb, params={"camera": "base_camera"})
    wrist_image = ObservationTermCfg(func=mdp.rgb, params={"camera": "wrist_camera"})
    concatenate_terms = False
    enable_corruption = False


@configclass
class CriticCfg(ObservationGroupCfg):
    state = ObservationTermCfg(func=mdp.critic_state, params={"asset_cfg": robot_selection()})
    concatenate_terms = True
    enable_corruption = False


@configclass
class ObservationsCfg:
    policy = PolicyCfg()
    critic = CriticCfg()


@configclass
class ActionsCfg:
    arm = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=True, ik_method="dls"
        ),
        scale=(0.02, 0.02, 0.02, 0.1, 0.1, 0.1),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
    )
    gripper = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint.*"],
        open_command_expr={"panda_finger_joint.*": 0.04},
        close_command_expr={"panda_finger_joint.*": 0.0},
    )


@configclass
class RobotSceneCfg:
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    base_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/BaseCamera",
        height=256,
        width=256,
        data_types=["rgb"],
        spawn=PinholeCameraCfg(
            focal_length=24.0, horizontal_aperture=30.0, clipping_range=(0.01, 20.0)
        ),
    )
    wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/WristCamera",
        update_latest_camera_pose=True,
        height=256,
        width=256,
        data_types=["rgb"],
        spawn=PinholeCameraCfg(
            focal_length=18.0, horizontal_aperture=30.0, clipping_range=(0.01, 20.0)
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.1, 0.0, 0.05),
            rot=(-0.00438587, -0.37390275, 0.19007240, 0.90777199),
            convention="ros",
        ),
    )


class Panda(EmbodimentBase):
    name = "panda"

    def __init__(self, config: ArenaLiberoConfig) -> None:
        super().__init__(enable_cameras=False)
        self.scene_config = RobotSceneCfg()
        robot = self.scene_config.robot
        robot.spawn.usd_path = config.panda_usd
        robot.init_state.pos = (2.432, -1.581, 0.75)
        robot.init_state.rot = (0.0, 0.0, -0.70710678, 0.70710678)
        robot.init_state.joint_pos = dict(
            zip(
                [f"panda_joint{i}" for i in range(1, 8)],
                [0.0, -0.569, 0.0, -2.81, 0.0, 3.037, 0.741],
                strict=True,
            )
        )
        robot.init_state.joint_pos["panda_finger_joint.*"] = 0.04
        for camera in (self.scene_config.base_camera, self.scene_config.wrist_camera):
            camera.height = config.image_size
            camera.width = config.image_size
        self.action_config = ActionsCfg()
        self.observation_config = ObservationsCfg()

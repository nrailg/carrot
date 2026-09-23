"""导入自己已准备好的机械臂 USD；不是包含 SO101 资产的开箱即用驱动。

官方教程（导入资产、编写机器人配置）：
https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/how-to/import_new_asset.html
https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/how-to/write_articulation_cfg.html
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


# [S3, S6] UsdFileCfg + ArticulationCfg + ImplicitActuatorCfg 是官方的资产配置方式。
# None 表示读取 USD 中已有的驱动参数，绝不把 Panda 的电机参数搬给小机械臂。
def make_custom_arm(usd_path: str, joint_defaults: dict[str, float]) -> ArticulationCfg:
    """Load a local fixed-base arm with position-drive parameters authored in its USD.

    Parameters
    ----------
    usd_path : str
        A complete USD asset with meshes, collisions, mass/inertia, joint limits and drives.
    joint_defaults : dict[str, float]
        Valid reset positions for every movable joint, including the gripper. Revolute
        joints use radians; prismatic joints use meters. Names must match the asset.

    Returns
    -------
    ArticulationCfg
        A robot configuration to pass to ``configure_robot`` before creating the environment.

    Raises
    ------
    FileNotFoundError
        The root USD file does not exist; referenced assets are checked by the simulator.
    """
    path = Path(usd_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(path),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos=joint_defaults),
        actuators={
            "arm_and_gripper": ImplicitActuatorCfg(
                joint_names_expr=[".*"], stiffness=None, damping=None
            ),
        },
    )

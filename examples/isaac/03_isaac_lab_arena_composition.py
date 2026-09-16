"""使用 Isaac Lab-Arena 将机器人、背景和物体组合成环境。"""

# ruff: noqa: E402, I001

from asset_root import configure_asset_root_from_env

configure_asset_root_from_env()

import torch

from isaaclab.app import AppLauncher
from isaaclab_arena.cli.isaaclab_arena_cli import (
    arena_env_builder_cfg_from_argparse,
    get_isaaclab_arena_cli_parser,
)


# Arena 和 Isaac Lab 共用一套启动参数，例如 --viz kit、--num_envs 和 --device。
parser = get_isaaclab_arena_cli_parser()
args_cli = parser.parse_args()

# 场景组件依赖 Isaac Sim 插件，因此要先启动应用，再导入下面的 Arena 模块。
simulation_app = AppLauncher(args_cli)

from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.utils.isaaclab_utils.simulation_app import teardown_simulation_app
from isaaclab_arena.utils.pose import Pose


def main() -> None:
    registry = AssetRegistry()

    # Embodiment 不只是机器人模型，还包含动作、观测、控制器和机载传感器。
    robot = registry.get_asset_by_name("franka_ik")(enable_cameras=args_cli.enable_cameras)

    # Scene 只描述物理世界。以后可以单独换机器人，而不重写这些场景资产。
    kitchen = registry.get_asset_by_name("kitchen")()
    cracker_box = registry.get_asset_by_name("cracker_box")()
    tomato_soup_can = registry.get_asset_by_name("tomato_soup_can")()
    light = registry.get_asset_by_name("light")()

    # 显式设置两个物体的位置，先把注意力放在组件组合，而不是随机摆放约束上。
    cracker_box.set_initial_pose(Pose(position_xyz=(0.4, -0.15, 0.1)))
    tomato_soup_can.set_initial_pose(Pose(position_xyz=(0.4, 0.15, 0.1)))
    scene = Scene(assets=[kitchen, cracker_box, tomato_soup_can, light])

    # 本例不设置 task，只演示 Arena 最核心的“独立组件再组合”思想。
    arena_environment = IsaacLabArenaEnvironment(
        name="franka_kitchen_beginner_example",
        embodiment=robot,
        scene=scene,
    )

    # Builder 把 Arena 组件编译成标准的 Isaac Lab ManagerBasedRLEnv。
    builder_cfg = arena_env_builder_cfg_from_argparse(args_cli)
    env = ArenaEnvBuilder(arena_environment, builder_cfg).make_registered()

    observation, _ = env.reset()
    print(f"并行环境数量: {env.unwrapped.num_envs}")
    print(f"动作空间: {env.action_space}")
    print(f"观测分组: {list(observation)}")

    # 零动作不会完成任务，但可以验证组合出的环境确实能够连续推进。
    for _ in range(60):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

    env.close()
    teardown_simulation_app(suppress_exceptions=False, make_new_stage=True)


if __name__ == "__main__":
    main()

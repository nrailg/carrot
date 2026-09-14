"""使用 Isaac Lab-Arena 将机器人、背景和物体组合成环境。"""

from isaaclab_arena.assets.asset_registry import AssetRegistry
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder, ArenaEnvBuilderCfg
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.scene.scene import Scene


def main() -> None:
    registry = AssetRegistry()

    # Embodiment 不只是机器人模型，还包含动作、观测、控制器和机载传感器。
    robot = registry.get_asset_by_name("franka_ik")(enable_cameras=False)

    # Scene 只描述物理世界。以后可以单独换机器人，而不重写这些场景资产。
    kitchen = registry.get_asset_by_name("kitchen")()
    cracker_box = registry.get_asset_by_name("cracker_box")()
    tomato_soup_can = registry.get_asset_by_name("tomato_soup_can")()
    scene = Scene(assets=[kitchen, cracker_box, tomato_soup_can])

    # 本例不设置 task，只演示 Arena 最核心的“独立组件再组合”思想。
    arena_environment = IsaacLabArenaEnvironment(
        name="franka_kitchen_beginner_example",
        embodiment=robot,
        scene=scene,
    )

    # Builder 把 Arena 组件编译成标准的 Isaac Lab ManagerBasedRLEnv。
    builder_cfg = ArenaEnvBuilderCfg(num_envs=4)
    env = ArenaEnvBuilder(arena_environment, builder_cfg).make_registered()

    observation, _ = env.reset()
    print(f"并行环境数量: {env.unwrapped.num_envs}")
    print(f"动作空间: {env.action_space}")
    print(f"观测分组: {list(observation)}")

    env.close()


if __name__ == "__main__":
    main()

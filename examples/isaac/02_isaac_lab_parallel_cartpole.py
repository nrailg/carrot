"""使用 Isaac Lab 在一个进程中并行运行多个 Cartpole 环境。"""

import argparse
import sys

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab_tasks.utils import (
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

parser = argparse.ArgumentParser(description="并行运行 Isaac Lab Cartpole 环境")
parser.add_argument("--task", default="Isaac-Cartpole-v0", help="Gymnasium 任务名称")
parser.add_argument("--num_envs", type=int, default=32, help="同一进程里的并行环境数量")
parser.add_argument("--num_steps", type=int, default=300, help="运行多少个控制步")
add_launcher_args(parser)

# 不传 --headless 时默认打开 Isaac Sim 窗口，便于观察复制出来的环境。
parser.set_defaults(visualizer=["kit"])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]


def main() -> None:
    torch.manual_seed(42)

    # 任务配置包含场景、观测、动作、奖励、终止条件和仿真参数。
    env_cfg, _ = resolve_task_config(args_cli.task, "")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        env_cfg.validate()

        env = gym.make(args_cli.task, cfg=env_cfg)
        observation, _ = env.reset()

        print(f"并行环境数量: {env.unwrapped.num_envs}")
        print(f"动作张量形状: {env.action_space.shape}")
        print(f"观测分组: {list(observation)}")

        for step in range(args_cli.num_steps):
            # action_space.shape 已包含环境维，因此一次生成全部环境的动作。
            actions = 2 * torch.rand(
                env.action_space.shape,
                device=env.unwrapped.device,
            ) - 1

            observation, rewards, terminated, truncated, _ = env.step(actions)

            if step % 50 == 0:
                finished = torch.count_nonzero(terminated | truncated).item()
                print(
                    f"step={step:3d}, "
                    f"平均奖励={rewards.mean().item():.3f}, "
                    f"本步结束环境数={finished}"
                )

        env.close()


if __name__ == "__main__":
    main()

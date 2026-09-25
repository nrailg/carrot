"""独立入门入口；python run.py --help。先读 README.md 和 SOURCES.md。

官方教程（启动顺序、reset/step 循环）：
https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/tutorials/00_sim/launch_app.html
https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/tutorials/03_envs/create_manager_rl_env.html
"""

# ruff: noqa: E402, I001

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher

# [S0] 官方要求先启动 AppLauncher，再导入依赖仿真插件的模块。
parser = argparse.ArgumentParser(description="自定义 reach 环境：先学 Lab，再看 Arena 组合。")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--policy", choices=("hold", "wiggle"), default="wiggle")
parser.add_argument(
    "--arena", action="store_true", help="用 Arena 组合相同任务并加入评测指标；默认直接用 Lab"
)
parser.add_argument("--robot_usd", type=Path, help="可选：同款 Panda 的本地 USD；不能用于换型号")
parser.add_argument("--asset_root", type=Path, help="含 Isaac/ 子目录的本地完整资产根目录")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.steps < 1:
    parser.error("--num_envs 和 --steps 必须为正整数")
if args.robot_usd is not None and not args.robot_usd.is_file():
    parser.error(f"USD 不存在：{args.robot_usd}")
launcher = AppLauncher(args)
simulation_app = launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from reach_env_cfg import ReachEnvCfg, configure_local_assets

# [S8] Arena 是可选的上层组合。普通 Lab 路径不导入或安装 Arena。
if args.arena:
    from arena_adapter import make_arena_env


def main() -> None:
    # [S2, S3] 配置是蓝图；实例化 ManagerBasedRLEnv 时才创建物理场景和 managers。
    cfg = ReachEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    if args.asset_root is not None:
        configure_local_assets(cfg, args.asset_root)
    if args.robot_usd is not None:
        cfg.scene.robot.spawn.usd_path = str(args.robot_usd.resolve())
    # SO101 的 configure_robot(...) 调用插在这里，详见 SO101.md。
    env = make_arena_env(cfg) if args.arena else ManagerBasedRLEnv(cfg=cfg)
    try:
        # [S2, S5] actions/observations 首维都是 N；一个进程推进 N 份物理世界。
        obs, _ = env.reset(seed=args.seed)
        robot = env.scene["robot"]
        print(f"joint_names={robot.joint_names}")
        print(f"body_names={robot.body_names}")
        print(
            f"policy={tuple(obs['policy'].shape)}, action={tuple(env.action_manager.action.shape)}"
        )
        print(f"control_dt={env.step_dt:.6f}s, max_episode_steps={env.max_episode_length}")
        terminated_count = 0
        truncated_count = 0
        for step in range(args.steps):
            if not simulation_app.is_running():
                break
            with torch.inference_mode():
                # 本例编写的演示输入：第一关节缓慢摆动，其他关节维持默认目标。
                # [S4] 0.5 的输入对应 0.125 rad 偏移；这不是解决 reach 的策略。
                action = torch.zeros_like(env.action_manager.action)
                if args.policy == "wiggle":
                    action[:, 0] = 0.5 * math.sin(2.0 * math.pi * 0.5 * step * env.step_dt)
                obs, reward, terminated, truncated, _ = env.step(action)
                # [S5] done 槽位已自动 reset；这里的 obs 属于它们的新回合。
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all(), (
                    "观测或奖励出现 NaN/Inf"
                )
                terminated_count += int(terminated.sum().item())
                truncated_count += int(truncated.sum().item())
                if (step + 1) % 30 == 0:
                    print(
                        f"step={step + 1}, reward_mean={reward.mean().item():.6f}, "
                        f"success_events={terminated_count}, timeout_events={truncated_count}"
                    )
        print("DEMO_FINISHED：环境循环结束；不代表策略完成了任务。")
    finally:
        env.close()


# [S0] 即使运行出错也关闭应用；异常继续向外传播，不伪装成成功。
if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

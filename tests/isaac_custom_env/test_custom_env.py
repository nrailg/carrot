"""真实 Isaac GPU 合同检查；通过同名 shell 启动，不作为 pytest 单测导入仿真器。

教程：https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/tutorials/03_envs/create_manager_rl_env.html
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path


def check_contracts(env: ManagerBasedRLEnv) -> dict:
    # 检查真实 PhysX 状态和批量 API；失败意味着环境不能按文档约定接收/产生张量。
    obs, _ = env.reset(seed=42)
    assert env.num_envs == 4
    assert obs["policy"].shape == (4, 28)
    assert env.action_manager.action.shape == (4, 7)
    assert math.isclose(env.step_dt, 1 / 30)
    assert env.max_episode_length == 150
    robot = env.scene["robot"]
    joint_ids, joint_names = robot.find_joints(ARM_JOINTS, preserve_order=True)
    assert joint_names == ARM_JOINTS
    body_cfg = SceneEntityCfg("robot", body_names=[EE_BODY])
    body_cfg.resolve(env.scene)
    assert len(body_cfg.body_ids) == 1

    # 先用不可达目标避免意外成功，真实推进物理让初始小扰动稳定。
    command = env.command_manager.get_command("ee_pose")
    action = torch.zeros_like(env.action_manager.action)
    for _ in range(30):
        command[:, :3] = 10.0
        obs, reward, terminated, truncated, _ = env.step(action)
        assert torch.isfinite(obs["policy"]).all()
        assert torch.isfinite(reward).all()
        assert not terminated.any() and not truncated.any()

    # 用实测刚体位置构造距离为零/很大的两种目标；不移动机器人，不冒充策略成功。
    hand_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        robot.data.body_pos_w.torch[:, body_cfg.body_ids[0]],
    )
    command[:, :3] = 10.0
    assert not reached_goal(env, "ee_pose", body_cfg, 0.03).any()
    command[0, :3] = hand_pos_b[0]
    success = reached_goal(env, "ee_pose", body_cfg, 0.03)
    assert success.tolist() == [True, False, False, False]

    # 检查动作顺序和量纲，确保 action 不是力矩或相对当前关节的增量。
    action[:, 0] = 0.1
    before = env.episode_length_buf.clone()
    obs, reward, terminated, truncated, _ = env.step(action)
    target = robot.data.joint_pos_target.torch[1:, joint_ids]
    expected = robot.data.default_joint_pos.torch[1:, joint_ids] + 0.25 * action[1:]
    torch.testing.assert_close(target, expected, atol=1e-6, rtol=0)

    # 只有槽位0成功并自动 reset；其他槽位的步数连续，奖励仍来自 reset 前状态。
    assert terminated.tolist() == [True, False, False, False]
    assert not truncated.any()
    assert env.episode_length_buf[0].item() == 0
    torch.testing.assert_close(env.episode_length_buf[1:], before[1:] + 1)
    assert reward[0] > 0.0
    assert (reward >= 0).all() and (reward <= env.step_dt + 1e-6).all()
    assert obs["policy"].shape == (4, 28)
    torch.testing.assert_close(obs["policy"][:, 14:21], command)
    assert (command[0, :3] < 1.0).all()
    assert (command[1:, :3] == 10.0).all()

    # 单独构造槽位1超时，证明 timeout 返回 truncated，且 reset 不影响其他槽位。
    command[:, :3] = 10.0
    env.episode_length_buf[1] = env.max_episode_length - 1
    before = env.episode_length_buf.clone()
    obs, reward, terminated, truncated, _ = env.step(action)
    assert not terminated.any()
    assert truncated.tolist() == [False, True, False, False]
    assert env.episode_length_buf[1].item() == 0
    keep = torch.tensor([0, 2, 3], device=env.device)
    torch.testing.assert_close(env.episode_length_buf[keep], before[keep] + 1)
    assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
    assert (command[1, :3] < 1.0).all()
    assert (command[keep, :3] == 10.0).all()

    # 恢复正常采样后连续走完一个完整回合长度，检查数值稳定与自然自动重置。
    obs, _ = env.reset(seed=43)
    done_events = 0
    for _ in range(160):
        obs, reward, terminated, truncated, _ = env.step(torch.zeros_like(action))
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
        done_events += int((terminated | truncated).sum().item())
    assert done_events >= 4
    return {
        "observation_shape": list(obs["policy"].shape),
        "action_shape": list(action.shape),
        "step_dt": env.step_dt,
        "max_episode_length": env.max_episode_length,
        "action_scale_and_order": "PASS",
        "controlled_success_and_partial_reset": "PASS",
        "controlled_timeout_and_partial_reset": "PASS",
        "post_reset_goal_observation": "PASS",
        "natural_rollout_steps": 160,
        "natural_done_events": done_events,
        "joint_names": robot.joint_names,
        "body_names": robot.body_names,
    }


def main() -> None:
    # 使用与入门入口相同的配置和环境构造，不用 fake backend 代替真实 GPU 仿真。
    cfg = ReachEnvCfg()
    cfg.scene.num_envs = 4
    cfg.sim.device = args.device
    cfg.seed = 42
    configure_local_assets(cfg, args.asset_root)
    env = make_arena_env(cfg) if args.arena else ManagerBasedRLEnv(cfg=cfg)
    try:
        with torch.inference_mode():
            result = check_contracts(env)
    finally:
        env.close()

    # 只有全部断言通过且环境已关闭才写 PASS；附实际文件哈希防止验证到旧版代码。
    files = sorted(example_dir.glob("*.py")) + [Path(__file__).resolve()]
    result.update(
        status="PASS",
        backend="arena" if args.arena else "lab",
        completed_at_utc=datetime.now(UTC).isoformat(),
        asset_root=str(args.asset_root.resolve()),
        python=sys.version,
        versions={
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions()
            if d.metadata["Name"].lower().replace("_", "-")
            in {"isaaclab", "isaaclab-arena", "isaacsim", "torch"}
        },
        sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"CUSTOM_ENV_CONTRACTS_PASS {args.output}", flush=True)


if __name__ == "__main__":
    # 在 AppLauncher 启动后才导入场景模块；普通 pytest 收集本文件不会启动仿真。
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arena", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not args.asset_root.is_dir():
        parser.error(f"Missing asset root: {args.asset_root}")
    launcher = AppLauncher(args)
    simulation_app = launcher.app

    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.math import subtract_frame_transforms

    example_dir = Path(__file__).resolve().parents[2] / "examples" / "isaac_custom_env"
    sys.path.insert(0, str(example_dir))
    from reach_env_cfg import (
        ARM_JOINTS,
        EE_BODY,
        ReachEnvCfg,
        configure_local_assets,
        reached_goal,
    )

    if args.arena:
        from arena_adapter import make_arena_env

    try:
        main()
    finally:
        simulation_app.close()

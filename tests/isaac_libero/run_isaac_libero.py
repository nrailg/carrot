"""Real Isaac rollout, reset semantics and PPO-update acceptance check."""

# ruff: noqa: E402

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=32)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 2 or args.steps < 16:
    parser.error("reset/PPO verification requires num_envs >= 2 and steps >= 16")
if not args.enable_cameras:
    parser.error("--enable_cameras is required")
launcher = AppLauncher(args)
simulation_app = launcher.app

import torch
from isaac_libero.rl_check import run_ppo_check
from isaaclab.utils.math import combine_frame_transforms
from PIL import Image

from carrot_sim.isaac_libero import make_isaac_libero_env
from carrot_sim.libero import IsaacLiberoConfig, LiberoVectorEnv


def verify_runtime(env: LiberoVectorEnv, output: Path) -> None:
    # 先执行控制和真实渲染，再检查模型输入的 shape、dtype 和可见像素。
    observation, _ = env.reset()
    initial_state = observation["policy"]["state"][0].tolist()
    initial_joints = env.backend.scene["robot"].data.joint_pos[0].tolist()
    initial_tcp = observation["policy"]["state"][:, :3].clone()
    action = torch.zeros((env.num_envs, 7), device=env.device)
    action[:, 6] = -1
    for _ in range(10):
        observation, *_ = env.step(action)
    assert observation["policy"]["state"].shape == (env.num_envs, 8)
    assert observation["critic"].shape == (env.num_envs, 52)
    assert torch.isfinite(observation["critic"]).all()
    robot = env.backend.scene["robot"]
    hand_id = robot.find_bodies("panda_hand")[0][0]
    camera = env.backend.scene["wrist_camera"]
    # USD pose caches can lag Fabric; derive the mounted pose from physics data.
    hand_pos = robot.data.body_pos_w[:1, hand_id]
    hand_quat = robot.data.body_quat_w[:1, hand_id]
    camera_pos, camera_quat = combine_frame_transforms(
        hand_pos,
        hand_quat,
        hand_pos.new_tensor([camera.cfg.offset.pos]),
        hand_pos.new_tensor([camera.cfg.offset.rot]),
    )
    poses = {
        "initial_state": initial_state,
        "initial_joints": initial_joints,
        "final_joints": robot.data.joint_pos[0].tolist(),
        "joint_targets": robot.data.joint_pos_target[0].tolist(),
        "hand_pos": robot.data.body_pos_w[0, hand_id].tolist(),
        "hand_quat": robot.data.body_quat_w[0, hand_id].tolist(),
        "camera_pos": camera_pos[0].tolist(),
        "camera_quat_ros": camera_quat[0].tolist(),
        "bowl_pos": env.backend.scene["akita_black_bowl"].data.root_pos_w[0].tolist(),
        "plate_pos": env.backend.scene["plate"].data.root_pos_w[0].tolist(),
    }
    (output / "camera_poses.json").write_text(json.dumps(poses, indent=2) + "\n")
    drift = (observation["policy"]["state"][:, :3] - initial_tcp).norm(dim=-1)
    assert drift.max() < 0.02, f"zero-action TCP drift: {drift.tolist()}"
    for name in ("image", "wrist_image"):
        rgb = observation["policy"][name]
        assert rgb.shape == (env.num_envs, 256, 256, 3) and rgb.dtype == torch.uint8
        assert (rgb.float().flatten(1).std(dim=1) > 1).all(), f"blank camera: {name}"
        Image.fromarray(rgb[0].cpu().numpy()).save(output / f"{name}.png")

    # 比较非目标槽的完整刚体与机器人状态，部分 reset 不应推进其物理状态。
    raw = env.backend
    untouched = torch.arange(1, env.num_envs, device=env.device)
    before = {
        name: raw.scene[name].data.root_state_w[untouched].clone()
        for name in ("robot", "akita_black_bowl", "plate")
    }
    before_joints = raw.scene["robot"].data.joint_pos[untouched].clone()
    before_age = raw.episode_length_buf[untouched].clone()
    env.reset(torch.tensor([0], device=env.device))
    for name, state in before.items():
        torch.testing.assert_close(
            raw.scene[name].data.root_state_w[untouched], state, rtol=0, atol=0
        )
    torch.testing.assert_close(
        raw.scene["robot"].data.joint_pos[untouched], before_joints, rtol=0, atol=0
    )
    torch.testing.assert_close(raw.episode_length_buf[untouched], before_age, rtol=0, atol=0)

    # 仅让槽 0 达到时间上限；确认最终观测在 autoreset 前保存且之后不会被覆盖。
    env.reset()
    raw.episode_length_buf[0] = raw.max_episode_length - 1
    obs, reward, terminated, truncated, info = env.step(action)
    assert truncated[0] and not terminated[0] and reward[0] == 0
    assert not (terminated | truncated)[1:].any()
    assert info["_final_observation"][0] and raw.episode_length_buf[0] == 0
    final_critic = info["final_observation"]["critic"].clone()
    env.step(action)
    torch.testing.assert_close(info["final_observation"]["critic"], final_critic, rtol=0, atol=0)
    assert torch.isfinite(obs["critic"]).all()

    # 构造满足上游几何判定的状态，验证真正的成功 term 到 reward/终止信号的链路。
    # 这是状态注入 fixture，不代表策略学会了搬运，也不改变 task 的成功函数。
    env.reset()
    selected = torch.tensor([0], device=env.device)
    target = raw.scene["robot"].data.root_pos_w[:1] + action.new_tensor([1.5, 0.0, 0.8])
    for name, height in (("plate", 0.0), ("akita_black_bowl", 0.08)):
        state = raw.scene[name].data.root_state_w[:1].clone()
        state[:, :3] = target + action.new_tensor([0.0, 0.0, height])
        state[:, 7:] = 0
        raw.scene[name].write_root_state_to_sim(state, env_ids=selected)
    raw.episode_length_buf[0] = 10
    obs, reward, terminated, truncated, info = env.step(action)
    assert terminated[0] and not truncated[0] and info["success"][0] and reward[0] == 1
    assert info["_final_observation"][0]
    assert (info["final_observation"]["critic"][0] - obs["critic"][0]).abs().max() > 0.1
    assert raw.cfg.isaaclab_arena_env.task._success_cache[0] == 0
    assert not raw.cfg.isaaclab_arena_env.task._success_flag[0]


def main() -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    config = IsaacLiberoConfig(num_envs=args.num_envs, device=args.device, max_episode_steps=16)
    env = make_isaac_libero_env(config)
    try:
        verify_runtime(env, args.output)
        # 使用真实视觉 rollout 更新网络；这是可训练性 smoke，不是收敛或成功率评测。
        torch.manual_seed(7)
        ppo = run_ppo_check(env, args.steps)
        assert ppo["truncated"] > 0, "rollout did not exercise timeout bootstrapping"
        result = {
            "status": "PASS",
            "task": env.task_name,
            "num_envs": env.num_envs,
            "steps": args.steps,
            "partial_reset": "PASS",
            "terminal_observation": "PASS",
            "success_state_fixture": "PASS",
            "ppo": ppo,
            "versions": {
                name: version(name) for name in ("torch", "isaacsim", "isaaclab", "gymnasium")
            },
        }
    finally:
        env.close()
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

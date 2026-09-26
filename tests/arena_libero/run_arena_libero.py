"""Run after activating Carrot's Python 3.12 environment."""

import argparse
import json
import platform
import sys
import traceback
from importlib.metadata import version
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--asset-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps", type=int, default=32)
parser.add_argument("--num-envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 32 or args.num_envs < 2 or not args.enable_cameras:
    parser.error("requires steps>=32, num-envs>=2 and --enable_cameras")
app = AppLauncher(args).app

import torch  # noqa: E402
from arena_libero.ppo import ppo_smoke  # noqa: E402
from PIL import Image  # noqa: E402

from carrot_sim.arena_libero.config import ArenaLiberoConfig  # noqa: E402
from carrot_sim.arena_libero.environment import make_env  # noqa: E402


def success_fixture(env) -> None:
    # 用真实接触验证成功项，避免把仅几何重叠的 mock 当物理验收。
    backend = env.backend
    old_duration = backend.cfg.episode_length_s
    backend.cfg.episode_length_s = 4.0
    try:
        env.reset(seed=42)
        ids = torch.tensor([0], dtype=torch.int32, device=env.device)
        bowl = backend.scene["bowl"]
        pose = bowl.data.root_pose_w.torch[ids].clone()
        pose[:, :3] = backend.scene["plate"].data.root_pos_w.torch[ids]
        pose[:, 2] += 0.055
        bowl.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        bowl.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(1, 6, device=env.device), env_ids=ids
        )
        action = torch.zeros(env.num_envs, 7, device=env.device)
        action[:, 6] = -1

        # 让碗在盘上自然落稳，必须由 manager 返回成功和正奖励。
        for _ in range(80):
            _, reward, terminated, _, info = env.step(action)
            if info["success"][0]:
                assert terminated[0] and reward[0] > 0
                assert not info["bootstrap_mask"][0]
                assert info["final_observation"] is not None
                return
        raise AssertionError("bowl placed above plate never produced a supported success")
    finally:
        backend.cfg.episode_length_s = old_duration


def main():
    # 输出目录必须独立，避免把历史 PASS 当成本轮结果。
    args.output.mkdir(parents=True, exist_ok=False)
    env = make_env(
        ArenaLiberoConfig(asset_root=args.asset_root, num_envs=args.num_envs, max_episode_steps=16)
    )
    try:
        obs = env.reset(seed=42)
        for name in ("image", "wrist_image"):
            Image.fromarray(obs[name][0].cpu().numpy()).save(args.output / f"reset_{name}.png")
        reset_wrist = obs["wrist_image"].clone()
        neutral = torch.zeros(args.num_envs, 7, device=env.device)
        neutral[:, 6] = -1
        for _ in range(5):
            obs, _, _, _, _ = env.step(neutral)
        robot = env.backend.scene["robot"]
        hand_id = robot.find_bodies("panda_hand")[0][0]
        camera = env.backend.scene["wrist_camera"]
        poses = {
            "hand_position": robot.data.body_pos_w.torch[0, hand_id].tolist(),
            "hand_quaternion_xyzw": robot.data.body_quat_w.torch[0, hand_id].tolist(),
            "sensor_reported_camera_position": camera.data.pos_w.torch[0].tolist(),
            "sensor_reported_camera_quaternion_world_xyzw": camera.data.quat_w_world.torch[
                0
            ].tolist(),
            "bowl_position": env.backend.scene["bowl"].data.root_pos_w.torch[0].tolist(),
            "plate_position": env.backend.scene["plate"].data.root_pos_w.torch[0].tolist(),
            "state": obs["state"][0].tolist(),
        }
        (args.output / "camera_poses.json").write_text(json.dumps(poses, indent=2))
        assert obs["state"].shape == (args.num_envs, 8)
        assert obs["critic"].shape == (args.num_envs, 52)
        assert torch.isfinite(obs["critic"]).all()
        for name in ("image", "wrist_image"):
            assert obs[name].dtype == torch.uint8
            assert obs[name].shape == (args.num_envs, 256, 256, 3)
            assert obs[name].float().std() > 5
            Image.fromarray(obs[name][0].cpu().numpy()).save(args.output / f"{name}.png")

        # 首帧应已反映关节初态，不能等训练采样几步后才出现正确视角。
        reset_wrist_mae = (reset_wrist.float() - obs["wrist_image"].float()).abs().mean().item()
        assert reset_wrist_mae < 20.0, f"stale initial wrist frame: {reset_wrist_mae}"

        # 移动手腕后立即 reset，同一种子下图像必须恢复原视角。
        move = neutral.clone()
        before_move = env.reset(seed=42)["state"]
        move[:, 2] = 0.8
        for _ in range(12):
            moved, _, _, _, _ = env.step(move)
        tcp_motion = (moved["state"][:, :3] - before_move[:, :3]).norm(dim=-1)
        control = {
            "before": before_move[0].tolist(),
            "after": moved["state"][0].tolist(),
            "tcp_motion": tcp_motion.tolist(),
            "joint_positions": robot.data.joint_pos.torch[0].tolist(),
            "joint_targets": robot.data.joint_pos_target.torch[0].tolist(),
        }
        (args.output / "control.json").write_text(json.dumps(control, indent=2))
        Image.fromarray(moved["wrist_image"][0].cpu().numpy()).save(
            args.output / "moved_wrist_image.png"
        )
        repeated = env.reset(seed=42)
        Image.fromarray(repeated["wrist_image"][0].cpu().numpy()).save(
            args.output / "repeat_reset_wrist_image.png"
        )
        repeat_reset_mae = (
            (reset_wrist.float() - repeated["wrist_image"].float()).abs().mean().item()
        )
        assert repeat_reset_mae < 20.0, f"stale reset wrist frame: {repeat_reset_mae}"

        # 控制必须实际改变物理状态，防止只有网络参数更新却没有机器人动作。
        assert (tcp_motion > 0.01).all(), f"TCP did not respond to actions: {control}"
        close_action = neutral.clone()
        close_action[:, 6] = 1
        for _ in range(8):
            closed, _, _, _, _ = env.step(close_action)
        finger_motion = repeated["state"][:, 6:].mean(-1) - closed["state"][:, 6:].mean(-1)
        assert (finger_motion > 0.01).all(), f"gripper did not close: {finger_motion}"
        env.reset(seed=42)

        # 只重置一个 slot，其他物体的物理状态不得变化。
        before = env.backend.scene["bowl"].data.root_pos_w.torch.clone()
        env.reset(torch.tensor([0], device=env.device))
        after = env.backend.scene["bowl"].data.root_pos_w.torch.clone()
        torch.testing.assert_close(before[1:], after[1:], atol=1e-6, rtol=0)
        assert not torch.equal(before[0], after[0])

        # 人为将一个 slot 推到时限边界，验证终止前观测和 reset 后观测分离。
        env.backend.episode_length_buf[0] = 15
        action = torch.zeros(args.num_envs, 7, device=env.device)
        action[:, 6] = -1
        _, _, term, trunc, info = env.step(action)
        assert trunc[0] and not term[0]
        assert info["final_mask"][0] and info["bootstrap_mask"][0]
        assert info["final_observation"] is not None
        assert env.backend.episode_length_buf[0] == 0

        # PPO 通过真实 rollout 更新参数；不把它解释为任务已学会。
        success_fixture(env)
        ppo = ppo_smoke(env, args.steps)
        result = {
            "status": "PASS",
            "num_envs": args.num_envs,
            "ppo": ppo,
            "success_fixture": "PASS",
            "tcp_motion_min": tcp_motion.min().item(),
            "finger_motion_min": finger_motion.min().item(),
            "reset_wrist_mae": reset_wrist_mae,
            "repeat_reset_wrist_mae": repeat_reset_mae,
            "partial_reset": "PASS",
            "terminal_observation": "PASS",
            "versions": {
                "python": platform.python_version(),
                **{
                    name: version(name)
                    for name in ("torch", "isaacsim", "isaaclab", "isaaclab-arena", "ray")
                },
            },
            "lw_imported": any(name.startswith("lw_benchhub") for name in sys.modules),
        }
        assert not result["lw_imported"]
    finally:
        env.close()
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        app.close()

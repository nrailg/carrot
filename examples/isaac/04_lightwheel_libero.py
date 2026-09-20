"""加载 Lightwheel-LIBERO 的碗与盘子任务，观察 Panda、RGB 和成功信号。"""

# ruff: noqa: E402

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=Path, default=Path("lightwheel-libero-output"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.steps < 1:
    parser.error("--num_envs 和 --steps 必须大于零")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms
from lw_benchhub.utils.env import ExecuteMode, parse_env_cfg
from PIL import Image


def main() -> None:
    cfg = parse_env_cfg(
        scene_backend="robocasa",
        task_backend="robocasa",
        scene_name="libero-1-1",
        robot_name="Panda",
        task_name="L90K1PutTheBlackBowlOnThePlate",
        robot_scale=1.0,
        execute_mode=ExecuteMode.EVAL,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=True,
        enable_cameras=args.enable_cameras,
        headless_mode=args.headless,
        seed=args.seed,
        sources=["objaverse", "lightwheel", "aigen_objs"],
        resample_objects_placement_on_reset=True,
        resample_robot_placement_on_reset=True,
    )
    # 上游 Panda embodiment 未预置 RGB 相机；相机随机器人基座放置。
    if args.enable_cameras:
        cfg.scene.example_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0/example_camera",
            update_period=0.0,
            height=224,
            width=224,
            data_types=["rgb"],
            spawn=PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 20.0)),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(1.0, 0.0, 0.8),
                rot=(0.65328, 0.27060, 0.27060, 0.65328),
                convention="opengl",
            ),
        )
    env = gym.make("L90K1PutTheBlackBowlOnThePlate", cfg=cfg)
    try:
        raw_env = env.unwrapped
        observation, _ = env.reset(seed=args.seed)
        print(f"观测分组: {list(observation)}", flush=True)
        print(f"动作维度: {raw_env.action_manager.total_action_dim}", flush=True)
        robot = raw_env.scene["robot"]
        if args.enable_cameras:
            # 用任务物体定位观察相机，避免固定朝向只拍到厨房墙面。
            bowl_pos = raw_env.scene["akita_black_bowl"].data.root_pos_w
            plate_pos = raw_env.scene["plate"].data.root_pos_w
            focus = (bowl_pos + plate_pos) * 0.5
            eye_offset = torch.tensor([0.0, -0.5, 0.85], device=raw_env.device)
            raw_env.scene["example_camera"].set_world_poses_from_view(
                eyes=focus + eye_offset, targets=focus
            )
        hand_ids, _ = robot.find_bodies("panda_hand")
        hand_id = hand_ids[0]
        tool_offset = torch.tensor([[0.0, 0.0, 0.107]], device=raw_env.device)
        succeeded = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
        completed_steps = 0
        with torch.inference_mode():
            for _ in range(args.steps):
                if not simulation_app.is_running():
                    break
                hand_pos, hand_quat = subtract_frame_transforms(
                    robot.data.root_pos_w,
                    robot.data.root_quat_w,
                    robot.data.body_pos_w[:, hand_id],
                    robot.data.body_quat_w[:, hand_id],
                )
                target_pos, target_quat = combine_frame_transforms(
                    hand_pos, hand_quat, tool_offset.expand(args.num_envs, -1)
                )
                # absolute IK 的零向量不是静止动作：保持当前末端目标，并打开夹爪。
                action = torch.cat(
                    [
                        target_pos,
                        target_quat,
                        torch.ones((args.num_envs, 1), device=raw_env.device),
                    ],
                    dim=-1,
                )
                observation, reward, terminated, truncated, _ = env.step(action)
                succeeded |= raw_env.termination_manager.get_term("success")
                completed_steps += 1
        args.output.mkdir(parents=True, exist_ok=True)
        if args.enable_cameras:
            rgb = raw_env.scene["example_camera"].data.output["rgb"]
            assert rgb.shape == (args.num_envs, 224, 224, 3), rgb.shape
            assert torch.isfinite(rgb).all(), "相机输出含非有限数值"
            pixels = rgb[0].detach().cpu().numpy().astype(np.uint8)
            assert pixels.max() > pixels.min(), "相机输出是单色，不能确认场景正确渲染"
            Image.fromarray(pixels).save(args.output / "camera.png")
        result = {
            "task": "L90K1PutTheBlackBowlOnThePlate",
            "robot": "Panda",
            "num_envs": args.num_envs,
            "steps": completed_steps,
            "seed": args.seed,
            "policy": "hold_current_pose",
            "success_seen": succeeded.cpu().tolist(),
            "camera_enabled": args.enable_cameras,
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        assert completed_steps == args.steps, "仿真提前退出"
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

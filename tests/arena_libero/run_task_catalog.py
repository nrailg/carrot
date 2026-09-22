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
parser.add_argument("--task-id", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
from arena_libero.ppo import ppo_smoke  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    combine_frame_transforms,
    subtract_frame_transforms,
)
from PIL import Image  # noqa: E402

from carrot_sim.arena_libero.config import ArenaLiberoConfig  # noqa: E402
from carrot_sim.arena_libero.environment import make_env  # noqa: E402
from carrot_sim.arena_libero.tasks import get_task  # noqa: E402


def set_goal_joint(env, spec, value: float) -> None:
    support = env.backend.scene[spec.goal.support]
    ids = torch.tensor([0], device=env.device, dtype=torch.int32)
    position = support.data.joint_pos.torch[ids].clone()
    position[:, support.find_joints(spec.goal.joint)[0][0]] = value
    support.write_joint_position_to_sim_index(position=position, env_ids=ids)
    support.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(position), env_ids=ids)
    env.backend.sim.forward()
    env.backend.scene.update(env.backend.physics_dt)


def success_fixture(env, spec) -> None:
    # 只构造 env 0 的成功状态，其他 slot 必须继续各自的 episode。
    env.reset(seed=42)
    backend = env.backend
    old_duration = backend.cfg.episode_length_s
    backend.cfg.episode_length_s = 6.0
    goal = spec.goal
    ids = torch.tensor([0], device=env.device, dtype=torch.int32)
    neutral = torch.zeros(env.num_envs, 7, device=env.device)
    neutral[:, 6] = -1
    try:
        if goal.kind != "on_plate":
            set_goal_joint(
                env, spec, goal.closed_position if goal.kind == "in_closed_drawer" else 0.8
            )
        support = backend.scene[goal.support].data
        if goal.support_body:
            body = backend.scene[goal.support].find_bodies(goal.support_body)[0][0]
            pos, quat = support.body_pos_w.torch[ids, body], support.body_quat_w.torch[ids, body]
        else:
            pos, quat = support.root_pos_w.torch[ids], support.root_quat_w.torch[ids]
        offset = pos.new_tensor(goal.center).reshape(1, 3)
        if goal.kind != "in_closed_drawer":
            offset[:, 2] += goal.target_half_size[2] + 0.018
        target_pos, target_quat = combine_frame_transforms(pos, quat, offset)
        target = backend.scene[goal.target]
        target.write_root_pose_to_sim_index(
            root_pose=torch.cat((target_pos, target_quat), -1), env_ids=ids
        )
        target.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(1, 6, device=env.device), env_ids=ids
        )

        # 接触/静止必须经真实 PhysX 达成，不能只直接调用纯数学谓词。
        for _ in range(120):
            _, reward, term, _, info = env.step(neutral)
            assert not info["success"][1:].any(), "success leaked to another environment"
            if info["success"][0]:
                assert term[0] and reward[0] > 0 and not info["bootstrap_mask"][0]
                assert info["final_observation"] is not None
                return
        raise AssertionError(f"Physical success fixture failed: {spec.task_id}")
    finally:
        backend.cfg.episode_length_s = old_duration


def main() -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    spec = get_task(args.task_id)
    env = make_env(
        ArenaLiberoConfig(
            asset_root=args.asset_root,
            task_id=spec.task_id,
            num_envs=4,
            max_episode_steps=16,
        )
    )
    try:
        # 首帧、维数及元信息必须对应选择的任务，而非旧单任务的固定配置。
        obs = env.reset(seed=42)
        assert env.task_name == spec.task_id and env.task_description == spec.language
        assert obs["critic"].shape == (4, spec.critic_size)
        assert torch.isfinite(obs["critic"]).all()
        for camera in ("image", "wrist_image"):
            assert obs[camera].shape == (4, 256, 256, 3)
            assert obs[camera].float().std() > 5
            Image.fromarray(obs[camera][0].cpu().numpy()).save(args.output / f"{camera}.png")
        initial = {
            obj.name: env.backend.scene[obj.name].data.root_pose_w.torch.tolist()
            for obj in spec.objects
        }
        joints = {
            fixture.name: env.backend.scene[fixture.name].data.joint_pos.torch.tolist()
            for fixture in spec.fixtures
        }
        (args.output / "initial_state.json").write_text(
            json.dumps({"objects": initial, "joints": joints}, indent=2)
        )

        for obj in spec.objects:
            if obj.joints:
                assert not env.backend.scene[obj.name].is_fixed_base, obj.name

        # 顶抽屉目标碗必须实际在已打开的上层内部，不能只修改任务名称。
        if "in_top_drawer" in spec.name:
            cabinet = env.backend.scene["cabinet"]
            body = cabinet.find_bodies("StorageFurniture136_Drawer001")[0][0]
            target = env.backend.scene["bowl_target"].data
            position, _ = subtract_frame_transforms(
                cabinet.data.body_pos_w.torch[:, body],
                cabinet.data.body_quat_w.torch[:, body],
                target.root_pos_w.torch,
            )
            centered = position - position.new_tensor([0.0, -0.012, 0.104])
            assert (centered.abs() < position.new_tensor([0.15, 0.125, 0.048])).all(), position
        if spec.goal.kind == "in_closed_drawer":
            cabinet = env.backend.scene[spec.goal.support]
            joint = cabinet.find_joints(spec.goal.joint)[0][0]
            assert (cabinet.data.joint_pos.torch[:, joint] > 0.10).all()
        if spec.goal.kind == "on_lit_stove":
            stove = env.backend.scene[spec.goal.support]
            joint = stove.find_joints(spec.goal.joint)[0][0]
            assert (stove.data.joint_pos.torch[:, joint].abs() < 0.01).all()

        # 局部 reset 必须保留其他 slot 的物体与关节状态。
        env.reset(torch.tensor([0], device=env.device))
        for obj in spec.objects:
            actual = env.backend.scene[obj.name].data.root_pose_w.torch
            torch.testing.assert_close(
                actual[1:], actual.new_tensor(initial[obj.name])[1:], atol=1e-6, rtol=0
            )
        for fixture in spec.fixtures:
            actual = env.backend.scene[fixture.name].data.joint_pos.torch
            torch.testing.assert_close(
                actual[1:], actual.new_tensor(joints[fixture.name])[1:], atol=1e-6, rtol=0
            )

        # 强制超时保留 terminal observation；成功判定应与超时分开。
        env.backend.episode_length_buf[0] = 15
        neutral = torch.zeros(4, 7, device=env.device)
        neutral[:, 6] = -1
        _, _, term, trunc, info = env.step(neutral)
        assert trunc[0] and not term[0] and info["bootstrap_mask"][0]
        assert info["final_observation"] is not None
        success_fixture(env, spec)
        ppo = ppo_smoke(env, 32)
        result = {
            "status": "PASS",
            "task_id": spec.task_id,
            "language": spec.language,
            "critic_size": spec.critic_size,
            "num_envs": 4,
            "ppo": ppo,
            "partial_reset": "PASS",
            "terminal_observation": "PASS",
            "success_fixture": "PASS",
            "versions": {
                "python": platform.python_version(),
                **{
                    name: version(name)
                    for name in ("isaacsim", "isaaclab", "isaaclab-arena", "torch")
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

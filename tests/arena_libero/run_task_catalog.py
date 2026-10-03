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
parser.add_argument("--panda-usd", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--task-id", required=True)
parser.add_argument("--physics-substeps", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
from arena_libero.ppo import ppo_smoke  # noqa: E402
from arena_libero.task_fixtures import success_fixture  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    subtract_frame_transforms,
)
from PIL import Image  # noqa: E402

from carrot_sim.arena_libero.config import ArenaLiberoConfig  # noqa: E402
from carrot_sim.arena_libero.environment import make_env  # noqa: E402
from carrot_sim.arena_libero.tasks import get_task  # noqa: E402


def main() -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    spec = get_task(args.task_id)
    config_kwargs = {}
    if args.panda_usd is not None:
        config_kwargs["panda_usd"] = str(args.panda_usd.resolve(strict=True))
    env = make_env(
        ArenaLiberoConfig(
            asset_root=args.asset_root,
            device=args.device,
            physics_substeps=args.physics_substeps,
            task_id=spec.task_id,
            num_envs=4,
            max_episode_steps=16,
            **config_kwargs,
        )
    )
    try:
        # 调整物理精度不能改变 RL 控制频率和 episode 的时间含义。
        assert abs(env.backend.step_dt - 0.02) < 1e-9
        assert abs(env.backend.physics_dt - 0.02 / args.physics_substeps) < 1e-9
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
        if (
            spec.name
            == "LS_pick_up_black_bowl_in_top_drawer_of_wooden_cabinet_and_place_it_on_plate"
        ):
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
        success_fixture(env, spec, args.output)
        ppo = ppo_smoke(env, 32)
        result = {
            "status": "PASS",
            "task_id": spec.task_id,
            "language": spec.language,
            "critic_size": spec.critic_size,
            "num_envs": 4,
            "physics_substeps": args.physics_substeps,
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
    except BaseException:
        # Kit 关闭可能返回 0；失败必须有独立持久化证据，不能仅依赖进程退出码。
        (args.output / "failure.json").write_text(
            json.dumps({"task_id": spec.task_id, "traceback": traceback.format_exc()}, indent=2)
        )
        raise
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

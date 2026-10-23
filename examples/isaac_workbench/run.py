# ruff: noqa: E402, I001

import argparse
import json
import os
import traceback
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from logic import DEFAULT_STROKES, reference_image
from tasks import CODING_TASKS, CodingTask

parser = argparse.ArgumentParser(
    description="Isaac robot workbench: batched Linux coding and painting"
)
parser.add_argument("--task", choices=("computer", "paint"), required=True)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--robot-usd", type=Path, required=True)
parser.add_argument("--guest-dir", type=Path)
parser.add_argument("--accelerator", choices=("kvm", "tcg"), default="kvm")
parser.add_argument(
    "--exercise", default="sum_two", help="comma-separated task IDs, cycled across envs"
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--policy", choices=("scripted", "hold"), default="scripted")
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--strokes", type=Path, help="expert strokes as normalized UV coordinate lists")
parser.add_argument(
    "--reference-image", type=Path, help="target image observed by the painting policy"
)
parser.add_argument(
    "--check", action="store_true", help="validate scripted success or hold inactivity"
)
parser.add_argument("--partial-reset-check", action="store_true")
args = parser.parse_args()
assert args.steps > 0 and args.num_envs > 0 and args.robot_usd.is_file()
assert args.task != "computer" or (args.guest_dir is not None and args.guest_dir.is_dir())
assert not args.output.exists(), f"use a fresh output directory: {args.output}"
assert not args.reference_image or args.policy != "scripted" or args.strokes, (
    "a custom reference needs expert --strokes for the scripted baseline"
)
exercise_names = args.exercise.split(",")
assert all(name in CODING_TASKS for name in exercise_names)
args.output.mkdir(parents=True)
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "active_gpu": args.gpu,
        "physics_gpu": args.gpu,
        "multi_gpu": False,
        "limit_cpu_threads": 16,
        "enable_crashreporter": False,
        "width": 1280,
        "height": 900,
        "extra_args": [
            "--/plugins/carb.tasking.plugin/threadCount=16",
            "--/telemetry/enableAnonymousData=false",
            "--/telemetry/enableSentry=false",
        ],
    }
)

from policy import ScriptedPolicy
from vector_env import VectorWorkbench


def save_episode(
    env: VectorWorkbench, i: int, task: CodingTask, success: bool, policy: ScriptedPolicy | None
) -> dict:
    bench = env.environments[i]
    output = args.output / f"env-{i:03d}"
    Image.fromarray(bench.rgb()).save(output / "after.png")
    result = {
        "env_id": i,
        "task": args.task,
        "exercise": task.name if args.task == "computer" else None,
        "success": bool(success),
        "policy": args.policy,
        "score": float(env.scores[i]),
        "control_steps": int(env.counts[i]),
        "physics_steps": bench.physics_step,
        "keyboard_events": list(bench.events),
        "max_normal_force": max(sample["normal_force"] for sample in bench.samples),
    }
    if env.vms is not None:
        guest = env.vms.guests[i]
        guest.screenshot().save(output / "screen.png")
        result.update(
            grade=guest.grade(),
            hid_events=list(guest.events),
            guest_keys_released=not guest.down,
            guest_log=str(guest.serial_path),
            episode=guest.episode,
            expected_keys=policy.expected_keys if policy else [],
        )
    else:
        bench.canvas.image.save(output / "canvas.png")
        result.update(
            canvas_iou=bench.canvas.score(env.references[i]),
            contact_samples=bench.canvas.contact_samples,
        )
    np.savez_compressed(
        output / "trajectory.npz",
        **{key: np.asarray([sample[key] for sample in bench.samples]) for key in bench.samples[0]},
    )
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def reset_and_verify(env: VectorWorkbench, i: int) -> dict:
    others = [j for j in range(env.num_envs) if j != i]
    before = {j: env.environments[j].robot.get_joint_positions().copy() for j in others}
    velocities = {j: env.environments[j].robot.get_joint_velocities().copy() for j in others}
    canvases = {j: env.environments[j].canvas.image.tobytes() for j in others}
    counts, scores, needs_reset = env.counts.copy(), env.scores.copy(), env.needs_reset.copy()
    if env.vms is not None:
        guests = {
            j: (
                env.vms.guests[j].process.pid,
                env.vms.guests[j].episode,
                env.vms.guests[j].screenshot().tobytes(),
            )
            for j in others
        }
        old_pid = env.vms.guests[i].process.pid
        old_episode = env.vms.guests[i].episode
    env.reset([i])
    for j in others:
        np.testing.assert_array_equal(env.environments[j].robot.get_joint_positions(), before[j])
        np.testing.assert_array_equal(
            env.environments[j].robot.get_joint_velocities(), velocities[j]
        )
        assert env.environments[j].canvas.image.tobytes() == canvases[j]
        assert (env.counts[j], env.scores[j], env.needs_reset[j]) == (
            counts[j],
            scores[j],
            needs_reset[j],
        )
    assert env.counts[i] == 0 and env.scores[i] == 0 and not env.needs_reset[i]
    assert not env.environments[i].switches.down.any()
    assert env.environments[i].canvas.contact_samples == 0
    assert np.asarray(env.environments[i].canvas.image).min() == 255
    if env.vms is not None:
        assert env.vms.guests[i].process.pid != old_pid
        assert env.vms.guests[i].episode != old_episode
        assert env.vms.guests[i].grade()["attempt"] == 0
        assert not env.vms.guests[i].down
        for j, expected in guests.items():
            guest = env.vms.guests[j]
            assert (guest.process.pid, guest.episode, guest.screenshot().tobytes()) == expected
    print(f"PARTIAL_RESET_PASS env={i} unaffected={others}", flush=True)
    return {"reset_env_id": i, "unaffected_env_ids": others, "passed": True}


def main() -> None:
    strokes = DEFAULT_STROKES if args.strokes is None else json.loads(args.strokes.read_text())
    target = reference_image(strokes)
    if args.reference_image:
        with Image.open(args.reference_image) as image:
            target = image.convert("RGB").resize((384, 384))
    tasks = [CODING_TASKS[exercise_names[i % len(exercise_names)]] for i in range(args.num_envs)]
    env = None
    writers = []
    results = [None] * args.num_envs
    reset_checks = []
    try:
        env = VectorWorkbench(
            args.task,
            args.robot_usd,
            [target] * args.num_envs,
            args.output,
            args.guest_dir,
            tasks,
            args.accelerator,
            args.steps,
            resolution=(640, 480),
        )
        policies = [
            ScriptedPolicy(bench, task.demonstration, strokes)
            for bench, task in zip(env.environments, tasks, strict=True)
        ]
        for i, bench in enumerate(env.environments):
            output = args.output / f"env-{i:03d}"
            output.mkdir()
            Image.fromarray(bench.rgb()).save(output / "before.png")
            if args.task == "paint":
                target.save(output / "reference.png")
            writers.append(imageio.get_writer(str(output / "rollout.mp4"), fps=15))
        for step in range(args.steps):
            assert simulation_app.is_running()
            actions = np.stack(
                [
                    policy.action()
                    if args.policy == "scripted" and results[i] is None
                    else bench.home
                    for i, (policy, bench) in enumerate(
                        zip(policies, env.environments, strict=True)
                    )
                ]
            )
            obs, reward, terminated, truncated, info = env.step(actions)
            assert np.isfinite(reward).all() and np.isfinite(obs["joint_positions"]).all()
            for i in range(args.num_envs):
                if results[i] is not None:
                    continue
                if step % 4 == 0:
                    writers[i].append_data(obs["rgb"][i])
                finished = (
                    terminated[i]
                    or truncated[i]
                    or (args.policy == "scripted" and policies[i].finished)
                )
                if finished or step == args.steps - 1:
                    results[i] = save_episode(env, i, tasks[i], terminated[i], policies[i])
                    print(
                        f"EPISODE_FINISHED env={i} success={results[i]['success']} "
                        f"score={results[i]['score']:.4f}",
                        flush=True,
                    )
            if all(result is not None for result in results):
                break
            for i in np.flatnonzero(env.needs_reset):
                reset_checks.append(reset_and_verify(env, int(i)))
            if step % 120 == 0:
                print(
                    f"step={step} scores={info['scores'].round(3).tolist()} "
                    f"contacts={[round(b.last_force, 3) for b in env.environments]}",
                    flush=True,
                )
        if args.partial_reset_check:
            assert args.num_envs >= 2
            reset_checks.append(reset_and_verify(env, 0))
        summary = {
            "task": args.task,
            "num_envs": args.num_envs,
            "results": results,
            "partial_reset_checks": reset_checks,
            "success": all(result["success"] for result in results),
        }
        (args.output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"WORKBENCH_RESULT success={summary['success']} envs={args.num_envs}", flush=True)
        if args.check:
            if args.policy == "hold":
                for result in results:
                    assert result["score"] == 0 and result["max_normal_force"] == 0
                    assert not result["keyboard_events"] and not result["success"]
                    if args.task == "computer":
                        assert result["grade"]["attempt"] == 0 and not result["hid_events"]
                    else:
                        assert result["contact_samples"] == 0
                return
            assert summary["success"], (
                "one or more demonstrations failed; see per-environment result.json"
            )
            for result in results:
                assert result["max_normal_force"] > 0.02
                if args.task == "computer":
                    assert result["guest_keys_released"]
                    assert [k for _, k, down in result["keyboard_events"] if down] == result[
                        "expected_keys"
                    ]
    finally:
        for writer in writers:
            writer.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        exit_code = 1
        traceback.print_exc()
        raise
    finally:
        simulation_app.close(exit_code=exit_code)

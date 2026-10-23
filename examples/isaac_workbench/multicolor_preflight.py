import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from multicolor_env import MulticolorEnv


def run(args: argparse.Namespace) -> None:
    """Exercise physical grasp/draw/return using the exact seven-dimensional learner action API.

    Parameters
    ----------
    args : argparse.Namespace
        Workbench CLI arguments. ``--check`` requires every slot to finish successfully.
    """
    environment = MulticolorEnv(
        str(args.robot_usd), args.num_envs, args.steps, args.seed, args.num_colors
    )
    output = Path(args.output)
    writer = imageio.get_writer(output / "rollout.mp4", fps=10)
    completed = {}
    try:
        if args.partial_reset_check:
            (output / "partial-reset.json").write_text(
                json.dumps(environment.check_reset(), indent=2) + "\n"
            )
        environment.reset(seeds=list(range(args.seed, args.seed + args.num_envs)))
        for step in range(args.steps):
            actions = (
                environment.expert_actions()
                if args.policy == "scripted"
                else np.zeros((args.num_envs, 1, 7), dtype=np.float32)
            )
            result = environment.step(actions)
            if step % 2 == 0:
                writer.append_data(result["observations"][0]["observation/image"])
            if step % 20 == 0:
                diagnostic = []
                for env, info in zip(environment.envs, result["info"], strict=True):
                    diagnostic.append(
                        {
                            **info,
                            "eef": env.eef_pose()[0].tolist(),
                            "pen_positions": (env.pens.get_world_poses()[0] - env.origin).tolist(),
                            "tip_positions": (env.tips.get_world_poses()[0] - env.origin).tolist(),
                            "finger_forces": env.fingers.get_contact_force_matrix(
                                dt=1 / 240
                            ).tolist(),
                            "fingers": env.robot.get_joint_positions()[0, 7:].tolist(),
                            "grip_command": env.grip_command,
                        }
                    )
                with (output / "trace.jsonl").open("a") as stream:
                    stream.write(json.dumps(diagnostic) + "\n")
            if step % 100 == 0:
                print(json.dumps({"step": step, "info": result["info"]}), flush=True)
                environment.snapshot(str(output / "latest"))
            done = np.flatnonzero(result["terminated"] | result["truncated"]).tolist()
            for i in done:
                if i not in completed:
                    completed[i] = result["info"][i]
                    env = environment.envs[i]
                    env.canvas.image.save(output / f"env-{i}-canvas.png")
            if len(completed) == args.num_envs:
                break
            if done:
                environment.reset(done, seeds=[args.seed + 10_000 + i for i in done])
        environment.snapshot(str(output / "final"))
        (output / "result.json").write_text(json.dumps(completed, indent=2) + "\n")
        if args.check:
            assert len(completed) == args.num_envs, "preflight ended before all episodes completed"
            if args.policy == "scripted":
                assert all(
                    row["success"]
                    and row["picks"] == args.num_colors
                    and row["returns"] == args.num_colors
                    and row["drops"] == 0
                    for row in completed.values()
                ), completed
            else:
                assert all(
                    row["contact_samples"] == 0 and row["iou"] == 0 for row in completed.values()
                ), completed
    finally:
        writer.close()
        environment.close()

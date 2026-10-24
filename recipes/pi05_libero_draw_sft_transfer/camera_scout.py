import argparse
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from carrot.trainer.rl.simulator import SimulatorWorker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--robot-usd", required=True)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--camera-config", type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    configurations = json.loads(args.candidates.read_text())
    camera = json.loads(args.camera_config.read_text()) if args.camera_config is not None else None
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "candidates.json").write_text(json.dumps(configurations, indent=2) + "\n")
    os.environ.update(LOCAL_RANK="0", MASTER_ADDR="127.0.0.1")
    simulator = SimulatorWorker(
        str(source / "examples/isaac_workbench/multicolor_server.py"),
        {
            "robot_usd": args.robot_usd,
            "num_envs": 1,
            "max_episode_steps": 4000,
            "num_colors": 3,
            "seed": 300000,
            "teacher_mode": "events",
            "camera": camera,
        },
        str(args.output / "renderer"),
        "127.0.0.1",
        python="/opt/venvs/carrot/bin/python",
    )
    simulator.setup()
    writer = imageio.get_writer(args.output / "camera-motion.mp4", fps=20)
    font = ImageFont.load_default(size=15)
    previous_phase = None
    snapshots = []
    states, action_rows = [], []
    try:
        observations = simulator.request("reset", seeds=[300000], num_colors=3)
        task_prompt = observations[0]["prompt"]
        for control in range(4000):
            phase = observations[0]["prompt"]
            assert phase == task_prompt, "actor received a changing subtask instruction"
            if control % 100 == 0 or phase != previous_phase:
                views = simulator.request("camera_views", configurations=configurations)
                sheet = Image.new("RGB", (3 * 224, 2 * 256), "#17202a")
                for index, (name, pixels) in enumerate(views.items()):
                    image = Image.fromarray(pixels)
                    image.save(args.output / f"{control:04d}-{name}.png")
                    x, y = (index % 3) * 224, (index // 3) * 256
                    sheet.paste(image, (x, y + 32))
                    ImageDraw.Draw(sheet).text((x + 4, y + 4), name, font=font)
                sheet.save(args.output / f"compare-{control:04d}.png")
                snapshots.append({"control": control, "prompt": phase})
                (args.output / "snapshots.json").write_text(json.dumps(snapshots, indent=2) + "\n")
                print(json.dumps({"snapshot": control, "prompt": phase}), flush=True)
                if args.preview_only:
                    break
            previous_phase = phase
            writer.append_data(observations[0]["observation/image"])
            states.append(observations[0]["observation/state"].copy())
            actions = simulator.request("expert_actions")
            action_rows.append(actions[0, 0].copy())
            result = simulator.request("step", actions=actions)
            observations = result["observations"]
            if result["terminated"][0] or result["truncated"][0]:
                (args.output / "result.json").write_text(
                    json.dumps(result["info"][0], indent=2) + "\n"
                )
                print(json.dumps(result["info"][0]), flush=True)
                info = result["info"][0]
                assert info["success"] and info["picks"] == info["returns"] == 3
                assert info["drops"] == 0
                break
    finally:
        writer.close()
        np.savez(
            args.output / "motion.npz", states=np.asarray(states), actions=np.asarray(action_rows)
        )
        simulator.teardown()


if __name__ == "__main__":
    main()

import argparse
import hashlib
import json
import os
import secrets
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from carrot.trainer.rl.simulator import SimulatorWorker

FPS = 20
FONT = ImageFont.load_default(size=15)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def frame(
    image: np.ndarray,
    wrist: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    step: int,
    phase: int,
    seed: int,
    mode: str,
) -> np.ndarray:
    canvas = Image.new("RGB", (448, 320), "#17202a")
    canvas.paste(Image.fromarray(image), (0, 72))
    canvas.paste(Image.fromarray(wrist), (224, 72))
    draw = ImageDraw.Draw(canvas)
    name = ("pick", "draw", "place")[phase % 3]
    quiet = np.linalg.norm(action[:3]) < 0.01
    draw.text((8, 5), f"{mode} | seed {seed} | {step / FPS:.2f}s | frame {step}", font=FONT)
    draw.text(
        (8, 27),
        f"stroke {phase // 3 + 1} / {name} | XYZ action {np.linalg.norm(action[:3]):.4f}",
        font=FONT,
    )
    draw.text(
        (8, 49),
        f"finger gap {np.abs(state[6:]).sum() * 1000:.1f}mm | grip {action[6]:+.0f}",
        font=FONT,
    )
    draw.text((8, 299), "BASE CAMERA", font=FONT)
    draw.text((232, 299), "WRIST CAMERA", font=FONT)
    if quiet:
        draw.rectangle((258, 48, 445, 69), fill="#ad6b00")
        draw.text((263, 49), "LOW XYZ COMMAND", font=FONT)
    return np.asarray(canvas)


def load_episode(path: Path) -> dict[str, np.ndarray]:
    return {
        name: np.load(path / f"{name}.npy", mmap_mode="r")
        for name in ("image", "wrist_image", "state", "actions", "phase")
    }


def pauses(data: dict[str, np.ndarray]) -> list[dict]:
    quiet = np.linalg.norm(data["actions"][:, :3], axis=1) < 0.01
    quiet[1:] &= np.linalg.norm(np.diff(data["state"][:, :3], axis=0), axis=1) < 0.0002
    edges = np.flatnonzero(np.diff(np.r_[False, quiet, False].astype(int)))
    return [
        {
            "start_seconds": int(start) / FPS,
            "end_seconds": int(end) / FPS,
            "duration_seconds": int(end - start) / FPS,
            "phase": int(data["phase"][start]),
        }
        for start, end in edges.reshape(-1, 2)
        if end - start >= FPS
    ]


def prepare(root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    population = sorted(root.glob("episode-*"))
    assert len(population) == 64
    sampling_seed = secrets.randbits(32)
    selected = np.random.default_rng(sampling_seed).choice(len(population), 5, replace=False)
    rows = []
    for index, choice in enumerate(selected):
        path = population[int(choice)]
        info = json.loads((path / "result.json").read_text())
        assert info["success"] and info["picks"] == info["returns"]
        data = load_episode(path)
        rows.append(
            {
                "index": index,
                "episode": path.name,
                "seed": info["seed"],
                "colors": info["picks"],
                "controls": len(data["actions"]),
                "actions_sha256": digest(path / "actions.npy"),
                "state_sha256": digest(path / "state.npy"),
                "near_stationary_intervals": pauses(data),
            }
        )
    selection = {
        "population": len(population),
        "sampling_seed": sampling_seed,
        "fps": FPS,
        "source": str(root),
        "episodes": rows,
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection), flush=True)
    for row in rows:
        data = load_episode(root / row["episode"])
        with imageio.get_writer(
            output / f"recorded-{row['index'] + 1}.mp4",
            fps=FPS,
            codec="libx264",
            ffmpeg_params=["-threads", "2"],
        ) as writer:
            for step, action in enumerate(data["actions"]):
                writer.append_data(
                    frame(
                        data["image"][step],
                        data["wrist_image"][step],
                        data["state"][step],
                        action,
                        step,
                        int(data["phase"][step]),
                        row["seed"],
                        "RECORDED",
                    )
                )
        print(json.dumps({"recorded_video_ready": row["index"] + 1}), flush=True)


def replay(output: Path, index: int, robot_usd: str, source: Path) -> None:
    selection = json.loads((output / "selection.json").read_text())
    row = selection["episodes"][index]
    path = Path(selection["source"]) / row["episode"]
    assert digest(path / "actions.npy") == row["actions_sha256"]
    data = load_episode(path)
    cpus = sorted(os.sched_getaffinity(0))[32 + 16 * index : 48 + 16 * index]
    assert len(cpus) == 16
    os.sched_setaffinity(0, cpus)
    os.environ.update(LOCAL_RANK="0", MASTER_ADDR="127.0.0.1")
    simulator = SimulatorWorker(
        str(source / "examples/isaac_workbench/multicolor_server.py"),
        {
            "robot_usd": robot_usd,
            "num_envs": 1,
            "max_episode_steps": 4000,
            "num_colors": row["colors"],
            "seed": row["seed"],
        },
        str(output / f"renderer-{index + 1}"),
        "127.0.0.1",
        python="/opt/venvs/carrot/bin/python",
    )
    simulator.setup()
    try:
        observations = simulator.request("reset", seeds=[row["seed"]], num_colors=row["colors"])
        with imageio.get_writer(
            output / f"isaac-{index + 1}.mp4",
            fps=FPS,
            codec="libx264",
            ffmpeg_params=["-threads", "2"],
        ) as writer:
            for step, action in enumerate(data["actions"]):
                obs = observations[0]
                writer.append_data(
                    frame(
                        obs["observation/image"],
                        obs["observation/wrist_image"],
                        obs["observation/state"],
                        action,
                        step,
                        int(data["phase"][step]),
                        row["seed"],
                        "ISAAC REPLAY",
                    )
                )
                result = simulator.request("step", actions=action[None, None])
                observations = result["observations"]
                if step % 200 == 0:
                    print(
                        json.dumps(
                            {
                                "replay": index + 1,
                                "step": step + 1,
                                "total": len(data["actions"]),
                                "info": result["info"][0],
                            }
                        ),
                        flush=True,
                    )
                if result["terminated"][0] or result["truncated"][0]:
                    break
        summary = {
            **row,
            "executed_controls": step + 1,
            "result": result["info"][0],
            "terminated": bool(result["terminated"][0]),
            "truncated": bool(result["truncated"][0]),
            "video": f"isaac-{index + 1}.mp4",
        }
        (output / f"replay-{index + 1}.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    finally:
        simulator.teardown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "replay"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--robot-usd")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        assert args.root is not None
        prepare(args.root, args.output)
    else:
        assert args.index is not None and 0 <= args.index < 5
        assert args.robot_usd is not None and args.source is not None
        replay(args.output, args.index, args.robot_usd, args.source)

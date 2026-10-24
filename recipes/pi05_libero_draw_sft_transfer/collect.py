import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from carrot.trainer.rl.simulator import SimulatorWorker


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def preflight(simulator: SimulatorWorker, output: Path) -> list[dict]:
    rows = []
    writer = imageio.get_writer(output / "teacher-preflight.mp4", fps=10)
    try:
        for start in range(300000, 300016, 4):
            simulator.request("reset", seeds=list(range(start, start + 4)), num_colors=3)
            completed = set()
            for control in range(4000):
                result = simulator.request("step", actions=simulator.request("expert_actions"))
                if control % 100 == 0:
                    row = {"preflight_start": start, "control": control + 1, "info": result["info"]}
                    with (output / "teacher-progress.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                    print(json.dumps(row), flush=True)
                if control % 2 == 0 and 0 not in completed:
                    writer.append_data(result["observations"][0]["observation/image"])
                done = np.flatnonzero(result["terminated"] | result["truncated"]).tolist()
                for slot in done:
                    if slot not in completed:
                        rows.append(result["info"][slot])
                        completed.add(slot)
                        print(json.dumps({"preflight_episode": rows[-1]}), flush=True)
                if len(completed) == 4:
                    break
                if done:
                    simulator.request("reset", ids=done, seeds=[400000 + start + i for i in done])
            assert len(completed) == 4, "teacher preflight incomplete"
            (output / "teacher-episodes.json").write_text(json.dumps(rows, indent=2) + "\n")
    finally:
        writer.close()
    assert len(rows) == 16 and all(
        row["success"] and row["picks"] == row["returns"] == 3 and row["drops"] == 0 for row in rows
    ), "event-driven teacher has not passed the physical gate"
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--robot-usd", required=True)
    parser.add_argument("--simulator-python", default="/opt/venvs/carrot/bin/python")
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--camera-config", type=Path)
    parser.add_argument("--reuse-physics-gate", type=Path)
    args = parser.parse_args()
    assert args.hours > 0
    source = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=False)
    camera = json.loads(args.camera_config.read_text()) if args.camera_config is not None else None
    gate = None
    if args.reuse_physics_gate is not None:
        gate = json.loads(args.reuse_physics_gate.read_text())
        assert gate["passed"] is True and gate["teacher_mode"] == "events"
        assert gate["task_contract"] == "static-reference-v1", "old oracle-guided task"
        assert gate["carrot_sha256"] == digest(
            source / "examples/isaac_workbench/multicolor_scene.py"
        )
        assert sorted(row["seed"] for row in gate["episodes"]) == list(range(300000, 300016))
        assert all(
            row["success"] and row["picks"] == row["returns"] == 3 and row["drops"] == 0
            for row in gate["episodes"]
        )
        gate["reused_physics_gate"] = {
            "path": str(args.reuse_physics_gate),
            "sha256": digest(args.reuse_physics_gate),
            "camera": gate.get("camera"),
            "scope": "unchanged event-driven teacher and physics; camera verified separately",
        }
        gate["camera"] = camera
    (args.output / "camera.json").write_text(json.dumps(camera, indent=2) + "\n")
    os.environ.update(LOCAL_RANK="0", MASTER_ADDR="127.0.0.1")
    simulator = SimulatorWorker(
        str(source / "examples/isaac_workbench/multicolor_server.py"),
        {
            "robot_usd": args.robot_usd,
            "num_envs": 4,
            "max_episode_steps": 4000,
            "num_colors": 3,
            "seed": 300000,
            "teacher_mode": "events",
            "camera": camera,
        },
        str(args.output / "renderer"),
        "127.0.0.1",
        python=args.simulator_python,
    )
    simulator.setup()
    try:
        simulator.request("check_reset")
        if gate is None:
            rows = preflight(simulator, args.output)
            gate = {
                "passed": True,
                "teacher_mode": "events",
                "task_contract": "static-reference-v1",
                "camera": camera,
                "episodes": rows,
                "carrot_sha256": digest(source / "examples/isaac_workbench/multicolor_scene.py"),
            }
        (args.output / "teacher-gate.json").write_text(json.dumps(gate, indent=2) + "\n")
        report = simulator.request(
            "collect",
            output=str(args.output / "demos"),
            successes=64,
            attempts=128,
            deadline=time.time() + args.hours * 3600,
            curriculum=False,
            start_seed=200000,
        )
        assert report["saved"] == 64, report
        files = []
        for path in sorted((args.output / "demos").rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "path": str(path.relative_to(args.output)),
                        "size": path.stat().st_size,
                        "sha256": digest(path),
                    }
                )
        (args.output / "manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
        print(
            json.dumps({"collection_complete": report["saved"], "output": str(args.output)}),
            flush=True,
        )
    finally:
        simulator.teardown()


if __name__ == "__main__":
    main()

import argparse
import json
import os
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from carrot.trainer.rl.simulator import SimulatorWorker


def evaluate(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "arguments.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    os.environ.update(LOCAL_RANK="0", MASTER_ADDR="127.0.0.1")
    assert 1 <= args.execute_steps <= 5
    policy = None
    replays = []
    if args.mode == "policy":
        assert args.host is not None and args.port is not None
        policy = WebsocketClientPolicy(host=args.host, port=args.port)
    elif args.mode == "replay":
        assert args.demonstrations is not None and args.execute_steps == 1
        by_seed = {}
        for path in Path(args.demonstrations).glob("episode-*"):
            info = json.loads((path / "result.json").read_text())
            by_seed[info["seed"]] = path
        replays = [np.load(by_seed[seed] / "actions.npy") for seed in args.seeds]
    metadata = policy.get_server_metadata() if policy is not None else {"mode": args.mode}
    (output / "policy.json").write_text(json.dumps(metadata, indent=2) + "\n")
    simulator = SimulatorWorker(
        args.simulator_script,
        {
            "robot_usd": args.robot_usd,
            "num_envs": len(args.seeds),
            "max_episode_steps": args.max_controls,
            "seed": 7,
            "num_colors": args.colors,
        },
        str(output / "renderer"),
        "127.0.0.1",
        python=args.simulator_python,
    )
    simulator.setup()
    writer = imageio.get_writer(output / "first-episode.mp4", fps=4)
    completed = {}
    started = time.monotonic()
    controls = 0
    inference_seconds = simulation_seconds = 0.0
    try:
        observations = simulator.request("reset", seeds=args.seeds, num_colors=args.colors)
        for chunk in range(args.max_controls):
            actions = np.zeros((len(args.seeds), args.execute_steps, 7), np.float32)
            predicted = None
            before = time.monotonic()
            if args.mode == "expert" or controls < args.teacher_prefix_controls:
                actions = simulator.request("expert_actions")
                if policy is not None and controls % 20 == 0:
                    predicted = [
                        policy.infer(o)["actions"][: args.execute_steps].tolist()
                        for o in observations
                    ]
            elif args.mode == "replay":
                for i, replay in enumerate(replays):
                    if i not in completed:
                        assert controls < len(replay), f"teacher ended before replay success: {i}"
                        actions[i, 0] = replay[controls]
            else:
                assert policy is not None
                for i, observation in enumerate(observations):
                    if i not in completed:
                        actions[i] = policy.infer(observation)["actions"][: args.execute_steps]
            inference_seconds += time.monotonic() - before
            states = [observation["observation/state"].tolist() for observation in observations]
            if chunk % 100 == 0:
                np.savez_compressed(
                    output / f"observations-{chunk:06d}.npz",
                    image=np.stack([o["observation/image"] for o in observations]),
                    wrist_image=np.stack([o["observation/wrist_image"] for o in observations]),
                    state=np.asarray(states),
                    prompt=np.asarray([o["prompt"] for o in observations]),
                )
            before = time.monotonic()
            result = simulator.request("step", actions=actions)
            simulation_seconds += time.monotonic() - before
            observations = result["observations"]
            controls += int(result["lengths"][0])
            with (output / "trace.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "controls": controls,
                            "states": states,
                            "actions": actions.tolist(),
                            "predicted_actions": predicted,
                            "info": result["info"],
                        }
                    )
                    + "\n"
                )
            if 0 not in completed:
                writer.append_data(observations[0]["observation/image"])
            done = np.flatnonzero(result["terminated"] | result["truncated"]).tolist()
            for i in done:
                if i not in completed:
                    completed[i] = {
                        **result["info"][i],
                        "terminated": bool(result["terminated"][i]),
                        "truncated": bool(result["truncated"][i]),
                    }
            if chunk % 20 == 0 or done:
                row = {
                    "chunk": chunk + 1,
                    "controls": controls,
                    "wall_seconds": time.monotonic() - started,
                    "inference_seconds": inference_seconds,
                    "simulation_seconds": simulation_seconds,
                    "completed": len(completed),
                    "active": [info for i, info in enumerate(result["info"]) if i not in completed],
                }
                with (output / "progress.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                (output / "episodes.json").write_text(json.dumps(completed, indent=2) + "\n")
            if len(completed) == len(args.seeds):
                break
            if done:
                observations = simulator.request(
                    "reset", ids=done, seeds=[2_000_000 + i for i in done], num_colors=args.colors
                )
        assert len(completed) == len(args.seeds), "incomplete closed-loop evaluation"
        simulator.request("snapshot", output=str(output / "final"))
        summary = {
            "episodes": len(completed),
            "successes": sum(x["success"] for x in completed.values()),
            "mean_iou": float(np.mean([x["iou"] for x in completed.values()])),
            "wall_seconds": time.monotonic() - started,
            "inference_seconds": inference_seconds,
            "simulation_seconds": simulation_seconds,
        }
        (output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    finally:
        writer.close()
        simulator.teardown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("policy", "expert", "replay"), default="policy")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--demonstrations")
    parser.add_argument("--execute-steps", type=int, default=5)
    parser.add_argument("--teacher-prefix-controls", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--simulator-script", required=True)
    parser.add_argument("--simulator-python", required=True)
    parser.add_argument("--robot-usd", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--colors", type=int, required=True)
    parser.add_argument("--max-controls", type=int, default=4000)
    evaluate(parser.parse_args())

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any

import grpc
import numpy as np
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import send_bytes_in_chunks

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
KEYS = tuple(f"{joint}.pos" for joint in JOINTS)
CALIBRATION_SHA = "68ba463158a5bfde211bb2633d552ba13790c17be378d2e657338990e3bf1243"


def predict(stub: Any, state: dict[str, float], step: int) -> tuple[np.ndarray, float]:
    raw = state | {"task": "Dance with the arm", "wrist": np.zeros((224, 224, 3), np.uint8)}
    observation = TimedObservation(time.time(), step, raw, must_go=True)
    start = time.perf_counter()
    chunks = send_bytes_in_chunks(pickle.dumps(observation), services_pb2.Observation)
    stub.SendObservations(chunks, timeout=10)
    response = stub.GetActions(services_pb2.Empty(), timeout=30)
    assert response.data, f"Official server returned no action at step {step}"
    actions = pickle.loads(response.data)
    assert len(actions) == 1 and actions[0].timestep == step
    action = actions[0].action.cpu().float().numpy()
    assert action.shape == (6,) and np.isfinite(action).all()
    return action, time.perf_counter() - start


def settle(robot: SOFollower, target: np.ndarray, timeout: float = 3.0) -> tuple[np.ndarray, bool]:
    deadline = time.perf_counter() + timeout
    while True:
        observation = robot.get_observation()
        actual = np.array([observation[key] for key in KEYS])
        reached = bool(np.max(np.abs(actual[:5] - target[:5])) <= 3.0)
        if reached or time.perf_counter() >= deadline:
            return actual, reached
        time.sleep(1 / 15)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--demonstration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=144)
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--reuse-loaded-server", action="store_true")
    parser.add_argument("--port", default="/dev/cu.usbmodem5C821078421")
    args = parser.parse_args()
    assert 1 <= args.steps <= 144
    args.output.mkdir(parents=True, exist_ok=False)
    demo = json.loads(args.demonstration.read_text())
    assert demo["action_names"] == list(KEYS)
    states, targets = np.array(demo["states"]), np.array(demo["actions"])
    assert states.shape == targets.shape == (144, 6)
    calibration_dir = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower"
    calibration_file = calibration_dir / "my_awesome_follower_arm.json"
    assert hashlib.sha256(calibration_file.read_bytes()).hexdigest() == CALIBRATION_SHA
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": list(KEYS)},
        "observation.images.wrist": {
            "dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channels"],
        },
    }
    specs = RemotePolicyConfig("pi05_control", args.checkpoint, features, 1, device="cuda")
    robot = SOFollower(SOFollowerRobotConfig(
        port=args.port, id="my_awesome_follower_arm", calibration_dir=calibration_dir,
        use_degrees=True, cameras={}, disable_torque_on_disconnect=False,
        max_relative_target=20.0, num_read_retries=2,
    ))
    rows = []
    channel = grpc.insecure_channel(args.server)
    try:
        grpc.channel_ready_future(channel).result(timeout=15)
        stub = services_pb2_grpc.AsyncInferenceStub(channel)
        stub.Ready(services_pb2.Empty(), timeout=10)
        if not args.reuse_loaded_server:
            stub.SendPolicyInstructions(
                services_pb2.PolicySetup(data=pickle.dumps(specs)), timeout=300,
            )
        for i in range(3):
            state = dict(zip(KEYS, states[i].tolist(), strict=True))
            action, latency = predict(stub, state, -3 + i)
            print(f"SMOKE {i}: action={action.tolist()} rpc_seconds={latency:.3f}", flush=True)
        if not args.play:
            (args.output / "metrics.json").write_text('{"status":"smoke_passed","frames":3}\n')
            return
        bus = robot.bus
        bus.connect()
        assert bus.is_calibrated, "Motor calibration differs from saved calibration"
        registers = {}
        for index, joint in enumerate(JOINTS):
            registers[joint] = {key: bus.read(key, joint, normalize=False, num_retry=2)
                                for key in ["P_Coefficient", "I_Coefficient", "D_Coefficient",
                                            "Torque_Enable", "Lock", "Status", "Operating_Mode"]}
            expected = {"P_Coefficient": 32 if index in (2, 3) else 16,
                        "I_Coefficient": 0, "D_Coefficient": 32, "Torque_Enable": 1,
                        "Lock": 1, "Status": 0, "Operating_Mode": 0}
            assert registers[joint] == expected, (joint, registers[joint], expected)
        (args.output / "registers_before.json").write_text(json.dumps(registers, indent=2) + "\n")
        limits = []
        for joint in JOINTS:
            cal = robot.calibration[joint]
            limits.append([bus._normalize({cal.id: cal.range_min + 1})[cal.id],
                           bus._normalize({cal.id: cal.range_max - 1})[cal.id]])
        lower, upper = np.array(limits).T
        initial_observation = robot.get_observation()
        initial = np.array([initial_observation[key] for key in KEYS])
        assert np.all(initial >= lower - 0.1) and np.all(initial <= upper + 0.1), initial
        goal = np.clip(states[0], lower, upper)
        for fraction in np.linspace(0, 1, 46):
            target = initial + fraction * (goal - initial)
            robot.send_action(dict(zip(KEYS, target.tolist(), strict=True)))
            time.sleep(1 / 15)
        actual, reached = settle(robot, goal, timeout=5)
        assert reached, f"Initial alignment failed: target={goal}, actual={actual}"
        print(f"ALIGNED {actual.tolist()}", flush=True)
        with (args.output / "trajectory.jsonl").open("w") as stream:
            for i in range(args.steps):
                start = time.perf_counter()
                observation = robot.get_observation()
                action, latency = predict(stub, observation, i)
                bounded = np.clip(action, lower, upper)
                sent = robot.send_action(dict(zip(KEYS, bounded.tolist(), strict=True)))
                sent_array = np.array([sent[key] for key in KEYS])
                actual, reached = settle(robot, sent_array)
                status = bus.sync_read("Status", normalize=False, num_retry=2)
                assert all(value == 0 for value in status.values()), status
                row = {"step": i, "state": [observation[key] for key in KEYS],
                       "prediction": action.tolist(), "sent": sent_array.tolist(),
                       "actual": actual.tolist(), "reached": reached, "status": status,
                       "rpc_seconds": latency, "seconds": time.perf_counter() - start}
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                rows.append(row)
                print(f"PLAY {i + 1}/{args.steps} reached={reached} "
                      f"actual={actual.round(2).tolist()} rpc={latency:.3f}s", flush=True)
                time.sleep(max(0, 1 / 15 - (time.perf_counter() - start)))
        actual = np.array([row["actual"] for row in rows])
        sent = np.array([row["sent"] for row in rows])
        predictions = np.array([row["prediction"] for row in rows])
        result = {"status": "completed", "frames": len(rows), "checkpoint": args.checkpoint,
                  "tracking_mae": np.abs(actual - sent).mean(axis=0).tolist(),
                  "prediction_demo_mae": (
                      np.abs(predictions - targets[:len(rows)]).mean(axis=0).tolist()
                  ),
                  "timeouts": sum(not row["reached"] for row in rows),
                  "rpc_mean_seconds": float(np.mean([row["rpc_seconds"] for row in rows]))}
        (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        if robot.bus.is_connected:
            try:
                raw = robot.bus.sync_read("Present_Position", normalize=False, num_retry=2)
                robot.bus.sync_write("Goal_Position", raw, normalize=False, num_retry=2)
                assert robot.bus.sync_read("Goal_Position", normalize=False, num_retry=2) == raw
                (args.output / "hold_raw.json").write_text(json.dumps(raw) + "\n")
                after = {joint: {key: robot.bus.read(key, joint, normalize=False, num_retry=2)
                                 for key in ["P_Coefficient", "I_Coefficient", "D_Coefficient",
                                             "Torque_Enable", "Lock", "Status"]}
                         for joint in JOINTS}
                (args.output / "registers_after.json").write_text(
                    json.dumps(after, indent=2) + "\n"
                )
            finally:
                robot.bus.disconnect(disable_torque=False)
        channel.close()


if __name__ == "__main__":
    main()

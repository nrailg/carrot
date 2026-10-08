"""Probe small fixed servo targets without a policy or PID changes."""

import argparse
import hashlib
import json
import signal
import time
from dataclasses import asdict
from pathlib import Path

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig


REGISTERS = (
    "Model_Number", "Operating_Mode", "Phase", "Torque_Enable", "P_Coefficient",
    "I_Coefficient", "D_Coefficient", "CW_Dead_Zone", "CCW_Dead_Zone",
    "Minimum_Startup_Force", "Torque_Limit", "Max_Torque_Limit", "Acceleration",
    "Goal_Time", "Goal_Velocity", "Protection_Current", "Goal_Position",
    "Present_Position", "Status", "Present_Voltage", "Present_Temperature",
)


def interrupted(signum: int, frame: object) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--p", type=int, choices=(16, 32), default=16)
    parser.add_argument("--i", type=int, choices=(0, 1), default=0)
    parser.add_argument("--gain-joint", choices=("both", "elbow_flex", "wrist_flex"),
                        default="both")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGTERM, interrupted)
    calibration_dir = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower"
    robot = SOFollower(SOFollowerRobotConfig(
        port="/dev/cu.usbmodem5C821078421", id="my_awesome_follower_arm",
        calibration_dir=calibration_dir, use_degrees=True, cameras={},
        disable_torque_on_disconnect=False,
    ))
    bus = robot.bus
    names = list(bus.motors)
    records = []
    armed = False
    completed = False
    saved_gains = {}
    with (args.output / "events.jsonl").open("x", buffering=1) as stream:
        def log(event: str, **values: object) -> None:
            row = {"event": event, "wall_time": time.time(),
                   "monotonic": time.monotonic(), **values}
            records.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")

        try:
            bus.connect()
            assert bus.is_calibrated, "hardware/file calibration mismatch"
            preflight = {key: bus.sync_read(key, normalize=False, num_retry=2)
                         for key in REGISTERS}
            assert all(v == 0 for v in preflight["Status"].values()), preflight
            assert all(v == 0 for v in preflight["Operating_Mode"].values()), preflight
            assert all(v & 0x10 == 0 for v in preflight["Phase"].values()), preflight
            assert all(v < 65 for v in preflight["Present_Temperature"].values()), preflight
            assert all(100 <= v <= 130 for v in preflight["Present_Voltage"].values()), preflight
            initial = preflight["Present_Position"]
            if args.reference:
                reference = json.loads(args.reference.read_text().splitlines()[0])
                assert reference["event"] == "preflight"
                assert reference["calibration"] == {
                    n: asdict(v) for n, v in bus.calibration.items()
                }, "reference calibration mismatch"
                initial = reference["registers"]["Present_Position"]
                assert all(abs(initial[n] - preflight["Present_Position"][n]) <= 170
                           for n in names), "reference pose too far from current pose"
            assert all(bus.calibration[n].range_min <= v <= bus.calibration[n].range_max
                       for n, v in initial.items()), initial
            log("preflight", registers=preflight,
                calibration={n: asdict(v) for n, v in bus.calibration.items()},
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            phases = [("initial_hold", "elbow_flex", 0.0),
                      ("elbow_minus4", "elbow_flex", -4.0),
                      ("elbow_return_from_below", "elbow_flex", 0.0),
                      ("elbow_plus4", "elbow_flex", 4.0),
                      ("elbow_return_from_above", "elbow_flex", 0.0),
                      ("wrist_minus3", "wrist_flex", -3.0),
                      ("wrist_return_from_below", "wrist_flex", 0.0),
                      ("wrist_plus3", "wrist_flex", 3.0),
                      ("wrist_return_from_above", "wrist_flex", 0.0)]
            targets = []
            for label, joint, delta in phases:
                target = dict(initial)
                resolution = bus.model_resolution_table[bus.motors[joint].model] - 1
                target[joint] += round(delta * resolution / 360)
                assert all(bus.calibration[n].range_min <= v <= bus.calibration[n].range_max
                           for n, v in target.items()), target
                assert abs(target[joint] - initial[joint]) * 360 / resolution <= 4.1
                targets.append((label, joint, target))
            log("plan", execute=args.execute, targets=targets, hold_seconds=4,
                p=args.p, i=args.i, gain_joint=args.gain_joint, reference=str(args.reference))
            if args.execute:
                # Latch all other axes once; re-commanding feedback each loop would move the goal.
                current = bus.sync_read("Present_Position", normalize=False, num_retry=2)
                bus.sync_write("Goal_Position", current, normalize=False, num_retry=2)
                assert bus.sync_read("Goal_Position", normalize=False, num_retry=2) == current
                armed = True
                bus.enable_torque(num_retry=2)
                gain_joints = ("elbow_flex", "wrist_flex") if args.gain_joint == "both" else (
                    args.gain_joint,
                )
                for joint in gain_joints:
                    original = {key: preflight[key][joint]
                                for key in ("P_Coefficient", "I_Coefficient")}
                    desired = {"P_Coefficient": args.p, "I_Coefficient": args.i}
                    if original != desired:
                        saved_gains[joint] = original
                        bus.write("Lock", joint, 0, normalize=False, num_retry=2)
                        for key, value in desired.items():
                            bus.write(key, joint, value, normalize=False, num_retry=2)
                            assert bus.read(key, joint, normalize=False, num_retry=2) == value
                        bus.write("Lock", joint, 1, normalize=False, num_retry=2)
                        log("temporary_gains", joint=joint, original=original, applied=desired)
                for label, joint, target in targets:
                    bus.sync_write("Goal_Position", target, normalize=False, num_retry=2)
                    assert bus.sync_read("Goal_Position", normalize=False, num_retry=2) == target
                    start = time.monotonic()
                    log("target", phase=label, joint=joint, raw=target)
                    while True:
                        pos = bus.sync_read("Present_Position", normalize=False, num_retry=2)
                        goal = bus.sync_read("Goal_Position", normalize=False, num_retry=2)
                        status = bus.sync_read("Status", normalize=False, num_retry=2)
                        telemetry = {key: bus.read(key, joint, normalize=False, num_retry=2)
                                     for key in ("Present_Velocity", "Present_Load",
                                                 "Present_Current", "Moving", "Goal_Position_2")}
                        elapsed = time.monotonic() - start
                        log("sample", phase=label, joint=joint, elapsed=elapsed,
                            raw_position=pos, raw_goal=goal, status=status, telemetry=telemetry)
                        assert goal == target, "goal changed without a local command"
                        assert all(v == 0 for v in status.values()), status
                        assert all(abs(pos[n] - initial[n]) <= 170 for n in names), pos
                        if elapsed >= 4:
                            break
                        time.sleep(.025)
                    temperature = bus.sync_read("Present_Temperature", normalize=False, num_retry=2)
                    log("phase_end", phase=label, temperature=temperature)
                    assert all(v < 65 for v in temperature.values()), temperature
                    motor_id = bus.motors[joint].id
                    actual = bus._normalize({motor_id: pos[joint]})[motor_id]
                    wanted = bus._normalize({motor_id: target[joint]})[motor_id]
                    print(label, f"goal={wanted:.6f} actual={actual:.6f} error={actual-wanted:+.6f}",
                          flush=True)
            completed = True
        finally:
            try:
                if bus.is_connected and armed:
                    for joint, original in saved_gains.items():
                        bus.write("Lock", joint, 0, normalize=False, num_retry=2)
                        for key, value in original.items():
                            bus.write(key, joint, value, normalize=False, num_retry=2)
                            assert bus.read(key, joint, normalize=False, num_retry=2) == value
                        bus.write("Lock", joint, 1, normalize=False, num_retry=2)
                        log("restored_gains", joint=joint, values=original)
                    raw = bus.sync_read("Present_Position", normalize=False, num_retry=2)
                    assert all(bus.calibration[n].range_min <= v <= bus.calibration[n].range_max
                               for n, v in raw.items()), raw
                    bus.sync_write("Goal_Position", raw, normalize=False, num_retry=2)
                    assert bus.sync_read("Goal_Position", normalize=False, num_retry=2) == raw
                    status = bus.sync_read("Status", normalize=False, num_retry=2)
                    log("exit_hold", raw=raw, status=status)
                    assert all(v == 0 for v in status.values()), status
            finally:
                if bus.is_connected:
                    bus.disconnect(disable_torque=False)
                (args.output / "summary.json").write_text(json.dumps({
                    "completed": completed, "execute": args.execute,
                    "targets": sum(r["event"] == "target" for r in records),
                    "samples": sum(r["event"] == "sample" for r in records),
                }, indent=2))


if __name__ == "__main__":
    main()

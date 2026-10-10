import argparse
import hashlib
import json
from pathlib import Path

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from live import CALIBRATION_SHA, JOINTS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", default="/dev/cu.usbmodem5C821078421")
    args = parser.parse_args()
    assert not args.output.exists()
    calibration_dir = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower"
    calibration = calibration_dir / "my_awesome_follower_arm.json"
    assert hashlib.sha256(calibration.read_bytes()).hexdigest() == CALIBRATION_SHA
    robot = SOFollower(SOFollowerRobotConfig(
        port=args.port, id="my_awesome_follower_arm", calibration_dir=calibration_dir,
        cameras={}, use_degrees=True, disable_torque_on_disconnect=False,
    ))
    bus = robot.bus
    registers = ["P_Coefficient", "I_Coefficient", "D_Coefficient", "Torque_Enable",
                 "Lock", "Status", "Operating_Mode"]
    try:
        bus.connect()
        assert bus.is_calibrated
        before = {joint: {key: bus.read(key, joint, normalize=False, num_retry=2)
                          for key in registers} for joint in JOINTS}
        for index, joint in enumerate(JOINTS):
            expected = {"P_Coefficient": 32 if index in (2, 3) else 16,
                        "I_Coefficient": 0, "D_Coefficient": 32, "Status": 0, "Operating_Mode": 0}
            assert all(before[joint][key] == value for key, value in expected.items()), before
            assert before[joint]["Torque_Enable"] in (0, 1) and before[joint]["Lock"] in (0, 1)
        raw = bus.sync_read("Present_Position", normalize=False, num_retry=2)
        assert all(robot.calibration[joint].range_min <= value <= robot.calibration[joint].range_max
                   for joint, value in raw.items()), raw
        bus.sync_write("Goal_Position", raw, normalize=False, num_retry=2)
        assert bus.sync_read("Goal_Position", normalize=False, num_retry=2) == raw
        bus.enable_torque(num_retry=2)
        after = {joint: {key: bus.read(key, joint, normalize=False, num_retry=2)
                         for key in registers} for joint in JOINTS}
        for joint in JOINTS:
            assert after[joint] == before[joint] | {"Torque_Enable": 1, "Lock": 1}, after
        args.output.write_text(json.dumps({"before": before, "after": after,
                                          "hold_raw": raw}, indent=2) + "\n")
        print(f"HOLD ENABLED, PID unchanged: {raw}", flush=True)
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()

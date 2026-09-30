"""Set one SO101 joint target."""
# python3 so101_set_joint.py --joint 3 --target 60

import argparse
import math
import time
from pathlib import Path

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint", type=int, choices=range(1, 7), required=True)
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument("--port", default="/dev/cu.usbmodem5C821078421")
    parser.add_argument("--robot-id", default="my_awesome_follower_arm")
    parser.add_argument("--calibration-dir", type=Path, default=(
        Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower"
    ))
    parser.add_argument("--observe-seconds", type=float, default=2.0)
    parser.add_argument("--max-delta", type=float, default=90.0)
    parser.add_argument(
        "--position-tolerance", type=float, default=1.0,
        help="allowed feedback overshoot",
    )
    args = parser.parse_args()
    assert math.isfinite(args.target), "target must be finite"
    assert math.isfinite(args.observe_seconds) and args.observe_seconds >= 0
    assert math.isfinite(args.max_delta) and args.max_delta > 0
    assert math.isfinite(args.position_tolerance) and args.position_tolerance >= 0
    assert (args.calibration_dir / f"{args.robot_id}.json").is_file(), "calibration missing"
    name = JOINTS[args.joint - 1]
    robot = SOFollower(SOFollowerRobotConfig(
        port=args.port, id=args.robot_id, calibration_dir=args.calibration_dir,
        use_degrees=True, cameras={}, disable_torque_on_disconnect=False,
    ))
    bus = robot.bus
    motor_id = bus.motors[name].id
    calibration = bus.calibration[name]
    target_raw = bus._unnormalize({motor_id: args.target})[motor_id]
    endpoints = bus._normalize({motor_id: calibration.range_min})[motor_id], (
        bus._normalize({motor_id: calibration.range_max})[motor_id]
    )
    assert min(endpoints) <= args.target <= max(endpoints), (
        f"target outside calibration: {min(endpoints):.3f} .. {max(endpoints):.3f}"
    )
    assert calibration.range_min <= target_raw <= calibration.range_max
    position_per_count = abs(endpoints[1] - endpoints[0]) / (
        calibration.range_max - calibration.range_min
    )
    raw_tolerance = args.position_tolerance / position_per_count
    try:
        bus.connect()
        assert bus.is_calibrated, "servo calibration differs from saved file"
        # Status 按位表示故障：1 电压、2 磁编码器、4 温度、8 电流、32 过载。
        # 0 表示无故障；多个故障可叠加，如 5=1+4；16、64、128 未定义。
        assert bus.read("Status", name, normalize=False, num_retry=2) == 0
        assert bus.read("Operating_Mode", name, normalize=False, num_retry=2) == 0
        assert bus.read("Phase", name, normalize=False, num_retry=2) & 0x10 == 0
        current_raw = bus.read("Present_Position", name, normalize=False, num_retry=2)
        assert (
            calibration.range_min - raw_tolerance <= current_raw
            <= calibration.range_max + raw_tolerance
        ), f"current position outside calibration by more than {args.position_tolerance}"
        current = bus._normalize({motor_id: current_raw})[motor_id]
        assert abs(args.target - current) <= args.max_delta + args.position_tolerance, (
            f"target delta exceeds {args.max_delta} + tolerance "
            f"{args.position_tolerance}; current={current:.3f}"
        )
        # 越界时只允许回到范围内，启用扭矩前的预置目标也须在范围内。
        hold_raw = min(calibration.range_max, max(calibration.range_min, current_raw))
        bus.write("Goal_Position", name, hold_raw, normalize=False, num_retry=2)
        bus.enable_torque(name, num_retry=2)
        bus.write("Goal_Position", name, target_raw, normalize=False, num_retry=2)
        assert bus.read("Goal_Position", name, normalize=False, num_retry=2) == target_raw
        print(f"Joint {args.joint} {name}: target={args.target:.3f}", flush=True)
        time.sleep(args.observe_seconds)
        actual = bus.read("Present_Position", name, num_retry=2)
        status = bus.read("Status", name, normalize=False, num_retry=2)
        print(f"Feedback={actual:.3f}, error={actual - args.target:+.3f}, status={status}")
        assert status == 0, f"servo status error: {status}"
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()

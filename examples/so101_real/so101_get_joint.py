"""Read SO101 joint angles in degrees, or gripper opening percentage, without moving."""

import argparse
from pathlib import Path

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint", type=int, choices=range(1, 7), help="omit to read all joints")
    parser.add_argument("--port", default="/dev/cu.usbmodem5C821078421")
    parser.add_argument("--robot-id", default="my_awesome_follower_arm")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower",
    )
    args = parser.parse_args()
    assert (args.calibration_dir / f"{args.robot_id}.json").is_file(), "calibration missing"
    robot = SOFollower(
        SOFollowerRobotConfig(
            port=args.port,
            id=args.robot_id,
            calibration_dir=args.calibration_dir,
            use_degrees=True,
            cameras={},
            disable_torque_on_disconnect=False,
        )
    )
    bus = robot.bus
    joint_ids = range(1, 7) if args.joint is None else (args.joint,)
    try:
        # Robot.connect() 会配置并启用扭矩；只连接 bus，保持读取操作没有运动副作用。
        bus.connect()
        assert bus.is_calibrated, "servo calibration differs from saved file"
        for joint_id in joint_ids:
            name = JOINTS[joint_id - 1]
            actual = bus.read("Present_Position", name, num_retry=2)
            # Status 按位表示故障：1 电压、2 磁编码器、4 温度、8 电流、32 过载。
            # 0 表示无故障；多个故障可叠加，如 5=1+4；16、64、128 未定义。
            status = bus.read("Status", name, normalize=False, num_retry=2)
            unit = "%" if name == "gripper" else "deg"
            print(f"Joint {joint_id} {name}: position={actual:.3f} {unit}, status={status}")
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()

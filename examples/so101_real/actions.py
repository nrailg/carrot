import time
from pathlib import Path
from typing import Protocol

import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from .config import JOINT_NAMES, DeploymentConfig
from .observations import Robot, joint_state


class ActionSink(Protocol):
    def check_initial(self, recorded_state: np.ndarray, *, multi_step: bool) -> dict: ...
    def send(self, action: np.ndarray) -> dict: ...


class LogSink:
    def check_initial(self, recorded_state: np.ndarray, *, multi_step: bool) -> dict:
        return {}

    def send(self, action: np.ndarray) -> dict:
        return {"executed": False, "target": action.tolist()}


class SO101Sink:
    """Apply calibrated limits to source targets.

    Parameters
    ----------
    robot : Robot
        Driver enforces max_relative_target.
    config : DeploymentConfig
        Initial tolerance and log arrays retain the source scale.
    lower, upper : np.ndarray
        Shape (6,); calibrated command limits.
    """

    def __init__(self, robot: Robot, config: DeploymentConfig,
                 lower: np.ndarray, upper: np.ndarray) -> None:
        self.robot = robot
        self.config = config
        self.lower = np.asarray(lower, dtype=np.float32).copy()
        self.upper = np.asarray(upper, dtype=np.float32).copy()
        assert self.lower.shape == self.upper.shape == (6,), "expected six action limits"
        assert np.isfinite(self.lower).all() and np.isfinite(self.upper).all(), "invalid limits"
        assert (self.lower < self.upper).all(), "empty action range"

    def check_initial(self, recorded_state: np.ndarray, *, multi_step: bool) -> dict:
        live = joint_state(self.robot.get_observation())
        difference = live - recorded_state
        # 连续执行数据集动作前要求实机起点接近录制起点。
        if multi_step:
            assert np.max(np.abs(difference)) <= self.config.initial_state_tolerance, (
                f"recorded/live initial state mismatch: {difference.tolist()}; "
                "align the robot manually or use a single-step run"
            )
        return {"initial_live_state": live.tolist(), "initial_difference": difference.tolist()}

    def send(self, action: np.ndarray) -> dict:
        assert action.shape == (6,) and np.isfinite(action).all(), "invalid SO101 action"
        present = joint_state(self.robot.get_observation())
        # 实际端点读数转float32可能位于向内取整的命令边界外一个ULP。
        assert ((present >= np.nextafter(self.lower, np.float32(-np.inf)))
                & (present <= np.nextafter(self.upper, np.float32(np.inf)))).all(), (
            "live joints are outside calibrated action limits; align the robot manually"
        )
        bounded = np.clip(action, self.lower, self.upper)
        # 绝对限幅在此执行，相对限幅由驱动执行。
        command = dict(zip(JOINT_NAMES, map(float, bounded), strict=True))
        sent = self.robot.send_action(command)
        # 驱动返回限幅后的发送目标，不代表运动后的实测位置。
        sent_targets = joint_state(sent)
        result = {
            "executed": True, "present": present.tolist(),
            "bounded_target": bounded.tolist(), "sent": sent_targets.tolist(),
            "absolute_clipped": (bounded != action).tolist(),
            "clip_delta": (bounded - action).tolist(),
            "clipped": not np.allclose(sent_targets, action, rtol=0, atol=1e-6),
        }
        if self.config.wait_for_target:
            result.update(self._wait_for_target(sent_targets))
        return result

    def _wait_for_target(self, target: np.ndarray) -> dict:
        start = time.monotonic()
        samples = []
        consecutive = 0
        while True:
            position = joint_state(self.robot.get_observation())
            elapsed = time.monotonic() - start
            error = position - target
            samples.append({"elapsed_s": elapsed, "position": position.tolist()})
            consecutive = consecutive + 1 if (
                np.abs(error) <= self.config.target_tolerance
            ).all() else 0
            # 连续读数达标，避免穿过目标的一瞬间就继续推理。
            reached = consecutive >= 3 and elapsed <= self.config.target_timeout_s
            if reached or elapsed >= self.config.target_timeout_s:
                return {"target_reached": reached, "target_error": error.tolist(),
                        "target_wait_s": elapsed, "target_samples": samples}
            time.sleep(min(0.02, self.config.target_timeout_s - elapsed))


def action_limits(robot: SO101Follower) -> tuple[np.ndarray, np.ndarray]:
    """Read calibrated command limits in the robot's configured action units.

    Parameters
    ----------
    robot : SO101Follower
        Uses its loaded LeRobot calibration; performs no hardware I/O.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Lower and upper float32 limits, shape (6,), in dataset joint order.
    """
    names = [name.removesuffix(".pos") for name in JOINT_NAMES]
    for name in names:
        calibration = robot.bus.calibration[name]
        assert calibration.range_min < calibration.range_max, f"invalid calibration: {name}"
    # 复用总线标定换算，避免硬编码动作范围。
    endpoints = [robot.bus._normalize({
        robot.bus.motors[name].id: (
            robot.bus.calibration[name].range_max if high else robot.bus.calibration[name].range_min
        ) for name in names
    }) for high in (False, True)]
    values = np.array([[point[robot.bus.motors[name].id] for name in names]
                       for point in endpoints], dtype=np.float64)
    lower, upper = values.min(axis=0), values.max(axis=0)
    lower32, upper32 = lower.astype(np.float32), upper.astype(np.float32)
    # float32边界向区间内取整，避免驱动转回整数刻度时越过标定端点。
    lower32 = np.where(lower32.astype(np.float64) < lower,
                       np.nextafter(lower32, np.float32(np.inf)), lower32)
    upper32 = np.where(upper32.astype(np.float64) > upper,
                       np.nextafter(upper32, np.float32(-np.inf)), upper32)
    return lower32, upper32


def load_action_limits(calibration_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load degree/gripper limits from an existing LeRobot SO101 calibration file.

    Parameters
    ----------
    calibration_path : str | pathlib.Path
        Shared calibration JSON; no serial port is opened or robot connected.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Five degree axes and one [0,100] gripper axis, as in action_limits.
    """
    path = Path(calibration_path).expanduser()
    assert path.is_file(), f"missing SO101 calibration: {path}"
    robot = SO101Follower(SO101FollowerConfig(
        port="unused", id=path.stem, calibration_dir=path.parent, use_degrees=True,
    ))
    return action_limits(robot)

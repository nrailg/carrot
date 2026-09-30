from typing import Protocol

import numpy as np

from .config import JOINT_NAMES
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
    """Send absolute joint targets bounded in the configured LeRobot units.

    Parameters
    ----------
    robot : Robot
        Driver must enforce max_relative_target in the same units as the policy.
    initial_tolerance : float
        Maximum per-joint recorded/live difference for multi-step dataset execution.
    lower, upper : np.ndarray
        Six limits derived from the driver's calibration and normalization modes.
    """

    def __init__(self, robot: Robot, initial_tolerance: float,
                 lower: np.ndarray, upper: np.ndarray) -> None:
        self.robot = robot
        self.initial_tolerance = initial_tolerance
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
            assert np.max(np.abs(difference)) <= self.initial_tolerance, (
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
        # 此处限制绝对目标范围；相对当前位置的变化限幅由 SO101 驱动执行。
        command = dict(zip(JOINT_NAMES, map(float, bounded), strict=True))
        sent = self.robot.send_action(command)
        actual = joint_state(sent)
        return {
            "executed": True, "present": present.tolist(),
            "bounded_target": bounded.tolist(), "sent": actual.tolist(),
            "clipped": bool(np.any(actual != action)),
        }

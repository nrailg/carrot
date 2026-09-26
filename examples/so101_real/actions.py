from typing import Protocol

import numpy as np

from .config import JOINT_NAMES, LOWER, UPPER
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
    """Send absolute normalized joint targets through a configured LeRobot driver.

    Parameters
    ----------
    robot : Robot
        Driver must use use_degrees=False and enforce max_relative_target itself.
    initial_tolerance : float
        Maximum per-joint recorded/live difference for multi-step dataset execution.
    """

    def __init__(self, robot: Robot, initial_tolerance: float) -> None:
        self.robot = robot
        self.initial_tolerance = initial_tolerance

    def check_initial(self, recorded_state: np.ndarray, *, multi_step: bool) -> dict:
        live = joint_state(self.robot.get_observation())
        difference = live - recorded_state
        if multi_step:
            assert np.max(np.abs(difference)) <= self.initial_tolerance, (
                f"recorded/live initial state mismatch: {difference.tolist()}; "
                "align the robot manually or use a single-step run"
            )
        return {"initial_live_state": live.tolist(), "initial_difference": difference.tolist()}

    def send(self, action: np.ndarray) -> dict:
        assert action.shape == (6,) and np.isfinite(action).all(), "invalid SO101 action"
        present = joint_state(self.robot.get_observation())
        bounded = np.clip(action, LOWER, UPPER)
        command = dict(zip(JOINT_NAMES, map(float, bounded), strict=True))
        sent = self.robot.send_action(command)
        actual = joint_state(sent)
        return {
            "executed": True, "present": present.tolist(),
            "bounded_target": bounded.tolist(), "sent": actual.tolist(),
            "clipped": bool(np.any(actual != action)),
        }

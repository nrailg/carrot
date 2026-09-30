from typing import Protocol

import numpy as np

from carrot.data.so101_units import to_model_units, to_robot_units

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
    """Convert model targets into driver units before applying calibrated limits.

    Parameters
    ----------
    robot : Robot
        Driver enforces max_relative_target in its native units.
    config : DeploymentConfig
        Initial tolerance is in driver units; returned log arrays use model units.
    lower, upper : np.ndarray
        Shape (6,); driver units, including gripper percentage points.
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
        recorded = to_robot_units(recorded_state, use_degrees=self.config.use_degrees)
        difference = live - recorded
        # 连续执行数据集动作前要求实机起点接近录制起点。
        if multi_step:
            assert np.max(np.abs(difference)) <= self.config.initial_state_tolerance, (
                f"recorded/live initial state mismatch: {difference.tolist()}; "
                "align the robot manually or use a single-step run"
            )
        live = to_model_units(live, use_degrees=self.config.use_degrees)
        difference = to_model_units(difference, use_degrees=self.config.use_degrees)
        return {"initial_live_state": live.tolist(), "initial_difference": difference.tolist()}

    def send(self, action: np.ndarray) -> dict:
        assert action.shape == (6,) and np.isfinite(action).all(), "invalid SO101 action"
        present = joint_state(self.robot.get_observation())
        # 实际端点读数转float32可能位于向内取整的命令边界外一个ULP。
        assert ((present >= np.nextafter(self.lower, np.float32(-np.inf)))
                & (present <= np.nextafter(self.upper, np.float32(np.inf)))).all(), (
            "live joints are outside calibrated action limits; align the robot manually"
        )
        target = to_robot_units(action, use_degrees=self.config.use_degrees)
        bounded = np.clip(target, self.lower, self.upper)
        # command使用驱动单位；绝对限幅在此执行，相对限幅由驱动执行。
        command = dict(zip(JOINT_NAMES, map(float, bounded), strict=True))
        sent = self.robot.send_action(command)
        # 驱动返回限幅后的发送目标；转回策略单位用于比较，不代表运动后的实测位置。
        sent_targets = joint_state(sent)
        present = to_model_units(present, use_degrees=self.config.use_degrees)
        bounded = to_model_units(bounded, use_degrees=self.config.use_degrees)
        sent_targets = to_model_units(sent_targets, use_degrees=self.config.use_degrees)
        return {
            "executed": True, "present": present.tolist(),
            "bounded_target": bounded.tolist(), "sent": sent_targets.tolist(),
            "clipped": not np.allclose(sent_targets, action, rtol=0, atol=1e-6),
        }

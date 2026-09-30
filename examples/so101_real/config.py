from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

JOINT_NAMES = (
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
)


@dataclass
class DeploymentConfig:
    observation_source: str = "dataset"
    action_sink: str = "log"
    server_uri: str = "ws://127.0.0.1:8000"
    output_dir: str = "outputs/so101_debug"
    dataset_root: str | None = None
    dataset_repo: str = ""
    base_camera: str | None = "top"
    wrist_camera: str = "fpv"
    episode: int = 0
    start_frame: int = 0
    prompt: str | None = None
    execute_steps: int = 1
    max_chunks: int | None = 1
    fps: int = 30
    connect_timeout_s: float = 10.0
    warmup_timeout_s: float = 60.0
    request_timeout_s: float = 5.0
    robot_port: str | None = None
    robot_id: str | None = None
    calibration_dir: str | None = None
    use_degrees: bool = True  # LeRobot驱动、录制数据和策略接口统一使用degree。
    max_relative_target: float = 5.0
    initial_state_tolerance: float = 10.0
    cameras: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def image_keys(self) -> dict[str, str]:
        keys = {"observation/wrist_image": self.wrist_camera}
        if self.base_camera is not None:
            keys["observation/image"] = self.base_camera
        return keys

    def validate(self) -> None:
        assert self.observation_source in ("dataset", "robot"), "invalid observation_source"
        assert self.action_sink in ("log", "robot"), "invalid action_sink"
        assert self.server_uri.startswith(("ws://", "wss://")), "server_uri must be ws(s)://"
        assert self.use_degrees is True, "SO101 recording and driver must use degrees"
        assert type(self.fps) is int and self.fps > 0, "FPS must be a positive integer"
        assert isinstance(self.wrist_camera, str) and self.wrist_camera.strip(), "set wrist_camera"
        assert self.base_camera is None or (
            isinstance(self.base_camera, str) and self.base_camera.strip()
            and self.base_camera != self.wrist_camera
        ), "base_camera must be distinct from wrist_camera, or null"
        for name, value in (("execute_steps", self.execute_steps), ("episode", self.episode),
                            ("start_frame", self.start_frame)):
            assert type(value) is int and value >= (1 if name == "execute_steps" else 0), name
        assert self.max_chunks is None or (type(self.max_chunks) is int and self.max_chunks > 0)
        for value in (self.connect_timeout_s, self.warmup_timeout_s, self.request_timeout_s,
                      self.max_relative_target, self.initial_state_tolerance):
            assert np.isfinite(value) and value > 0, "timeouts and motion limits must be positive"
        assert self.prompt is None or (isinstance(self.prompt, str) and self.prompt.strip())

        # 数据集回放与实时采集需要的输入资源不同。
        if self.observation_source == "dataset":
            assert self.dataset_root and Path(self.dataset_root).is_dir(), "set dataset_root"
            assert isinstance(self.dataset_repo, str) and self.dataset_repo.strip(), (
                "set dataset_repo explicitly"
            )
        else:
            assert self.prompt, "robot observations require a task prompt"
            assert set(self.cameras) == set(self.image_keys.values()), (
                "configure exactly the selected camera views"
            )
        if self.observation_source == "robot" or self.action_sink == "robot":
            assert self.robot_port and self.robot_id and self.calibration_dir, (
                "set robot_port, robot_id and calibration_dir"
            )
            assert (Path(self.calibration_dir) / f"{self.robot_id}.json").is_file(), (
                "existing robot calibration is required"
            )


def load_config(path: Path, overrides: dict[str, Any]) -> DeploymentConfig:
    with path.open() as stream:
        values = yaml.safe_load(stream)
    assert isinstance(values, dict), "deployment config must be a YAML mapping"
    assert values.keys() <= DeploymentConfig.__dataclass_fields__.keys(), "unknown config keys"
    # argparse 未提供的值为 None，只有显式传入的 CLI 参数才覆盖 YAML。
    values.update({key: value for key, value in overrides.items() if value is not None})
    config = DeploymentConfig(**values)
    config.validate()
    return config

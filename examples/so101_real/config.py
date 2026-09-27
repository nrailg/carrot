from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

JOINT_NAMES = (
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
)
# 此部署配置按 LeRobot 归一化位置约定限幅，夹爪下界与其他关节不同。
LOWER = np.array([-100, -100, -100, -100, -100, 0], dtype=np.float32)
UPPER = np.full(6, 100, dtype=np.float32)


@dataclass
class DeploymentConfig:
    observation_source: str = "dataset"
    action_sink: str = "log"
    server_uri: str = "ws://127.0.0.1:8000"
    output_dir: str = "outputs/so101_debug"
    dataset_root: str | None = None
    dataset_repo: str = "felixmayor/orange_cube_merged"
    dataset_revision: str = "c021b3c22a3de4e70e81010e54fb250a5dde348b"
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
    use_degrees: bool = False
    max_relative_target: float = 5.0
    initial_state_tolerance: float = 10.0
    cameras: dict[str, dict[str, Any]] = field(default_factory=dict)

    def validate(self) -> None:
        assert self.observation_source in ("dataset", "robot"), "invalid observation_source"
        assert self.action_sink in ("log", "robot"), "invalid action_sink"
        assert self.server_uri.startswith(("ws://", "wss://")), "server_uri must be ws(s)://"
        assert self.use_degrees is False, "this deployment profile requires use_degrees=false"
        assert self.fps == 30, "SO101 deployment profile requires 30 FPS"
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
            assert self.dataset_revision, "set the locally downloaded dataset revision"
        else:
            assert self.prompt, "robot observations require a task prompt"
            assert set(self.cameras) == {"top", "fpv"}, "configure top and fpv cameras"
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

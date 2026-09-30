from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .config import JOINT_NAMES, DeploymentConfig


class Robot(Protocol):
    def get_observation(self) -> dict[str, Any]: ...
    def send_action(self, action: dict[str, float]) -> dict[str, float]: ...


class Dataset(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> dict[str, Any]: ...


def joint_state(observation: dict[str, Any]) -> np.ndarray:
    state = np.array([observation[name] for name in JOINT_NAMES], dtype=np.float32)
    assert state.shape == (6,) and np.isfinite(state).all(), "invalid joint state"
    return state


def rgb_image(value: Any) -> np.ndarray:
    image = np.asarray(value)
    assert image.dtype == np.uint8 and image.ndim == 3, "expected RGB uint8 image"
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    assert image.shape[-1] == 3 and min(image.shape[:2]) > 0, "invalid RGB shape"
    return image.copy()


@dataclass
class ObservationFrame:
    request: dict[str, Any]
    episode: int | None
    frame: int
    reference: np.ndarray | None
    valid_steps: int


class ObservationSource(Protocol):
    def read(self) -> ObservationFrame | None: ...
    def advance(self, steps: int) -> None: ...


class DatasetSource:
    """Read one selected episode without exposing demonstration actions to the policy.

    Parameters
    ----------
    dataset : Dataset
        Must contain only ``episode`` and horizon-sized action chunks with padding masks.
    episode : int
    start_frame : int
        Episode-local frame index.
    horizon : int
    prompt : str | None
        Explicit override of the recorded task.
    image_keys : dict[str, str] | None
        Policy image fields mapped to dataset camera names; None uses top/fpv.
    """

    def __init__(self, dataset: Dataset, episode: int, start_frame: int, horizon: int,
                 prompt: str | None = None, *, image_keys: dict[str, str] | None = None) -> None:
        assert 0 <= start_frame < len(dataset), "start_frame outside selected episode"
        self.dataset = dataset
        self.episode = episode
        self.frame = start_frame
        self.horizon = horizon
        self.prompt = prompt
        self.image_keys = DeploymentConfig().image_keys if image_keys is None else image_keys.copy()

    def read(self) -> ObservationFrame | None:
        if self.frame >= len(self.dataset):
            return None
        sample = self.dataset[self.frame]
        assert int(sample["episode_index"]) == self.episode, "dataset crossed episode boundary"
        assert int(sample["frame_index"]) == self.frame, "unexpected dataset frame index"
        state = np.asarray(sample["observation.state"], dtype=np.float32)
        assert state.shape == (6,) and np.isfinite(state).all(), "invalid dataset state"
        reference = np.asarray(sample["action"], dtype=np.float32)
        padding = np.asarray(sample["action_is_pad"], dtype=bool)
        assert reference.shape == (self.horizon, 6), "unexpected demonstration action shape"
        assert np.isfinite(reference).all(), "non-finite demonstration"
        assert padding.shape == (self.horizon,), "invalid action padding mask"
        # 以 episode 剩余帧数核对 padding，避免跨边界执行预测动作。
        valid = min(self.horizon, len(self.dataset) - self.frame)
        assert np.array_equal(padding, np.arange(self.horizon) >= valid), (
            "action padding must match the end of the selected episode"
        )
        prompt = sample["task"] if self.prompt is None else self.prompt
        assert isinstance(prompt, str) and prompt.strip(), "missing dataset task"
        # 示教动作留在 reference，策略请求只包含观测和任务文本。
        return ObservationFrame(
            request={
                "observation/state": state.copy(),
                **{key: rgb_image(sample[f"observation.images.{camera}"])
                   for key, camera in self.image_keys.items()},
                "prompt": prompt,
            },
            episode=self.episode, frame=self.frame,
            reference=reference[:valid].copy(), valid_steps=valid,
        )

    def advance(self, steps: int) -> None:
        assert 0 < steps <= min(self.horizon, len(self.dataset) - self.frame)
        self.frame += steps


class RobotSource:
    def __init__(self, robot: Robot, prompt: str, horizon: int, *,
                 image_keys: dict[str, str] | None = None) -> None:
        self.robot = robot
        self.prompt = prompt
        self.horizon = horizon
        self.frame = 0
        self.image_keys = DeploymentConfig().image_keys if image_keys is None else image_keys.copy()

    def read(self) -> ObservationFrame:
        obs = self.robot.get_observation()
        return ObservationFrame(
            request={
                "observation/state": joint_state(obs),
                **{key: rgb_image(obs[camera]) for key, camera in self.image_keys.items()},
                "prompt": self.prompt,
            },
            episode=None, frame=self.frame, reference=None, valid_steps=self.horizon,
        )

    def advance(self, steps: int) -> None:
        self.frame += steps

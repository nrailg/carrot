"""SO101 LeRobot samples mapped to the PI0.5 transform contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from torch.utils.data import Dataset

from .dataset_spec import SFTDatasetSpec
from .so101_units import model_stats, to_model_units


class SO101SFTDataset(Dataset[dict[str, Any]]):
    """Expose absolute targets in model units and explicitly configured camera views.

    Parameters
    ----------
    source : LeRobotDataset
    base_image_key, wrist_image_key : str | None
        Omit a view with None; at least one camera must be explicitly configured.
    recorded_in_degrees : bool
        Convert degree angles to radians when true. Gripper percentages always
        become [0, 1] fractions; source samples are not modified.
    """

    def __init__(
        self, source: LeRobotDataset, *,
        base_image_key: str | None = None,
        wrist_image_key: str | None = None,
        recorded_in_degrees: bool = False,
    ) -> None:
        self.source = source
        self.recorded_in_degrees = recorded_in_degrees
        self.image_keys = {
            target: key for target, key in (
                ("observation/image", base_image_key),
                ("observation/wrist_image", wrist_image_key),
            ) if key is not None
        }
        assert self.image_keys, "SO101 requires at least one camera"

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.source[index]
        state, actions = sample["observation.state"], sample["action"]
        state = to_model_units(np.asarray(state), use_degrees=self.recorded_in_degrees)
        actions = to_model_units(np.asarray(actions), use_degrees=self.recorded_in_degrees)
        return {
            "observation/state": state,
            **{target: sample[key] for target, key in self.image_keys.items()},
            "prompt": sample["task"],
            "actions": actions,
            "action_is_pad": sample["action_is_pad"],
        }


def build_dataset(
    *,
    repo_id: str = "",
    root: str | None = None,
    revision: str | None = None,
    action_horizon: int = 50,
    video_backend: str | None = None,
    base_image_key: str | None = None,
    wrist_image_key: str | None = None,
    recorded_in_degrees: bool = False,
) -> SFTDatasetSpec:
    """Load SO101 LeRobot v3 data with episode-bounded future actions.

    Parameters
    ----------
    repo_id : str
        Explicit dataset identifier; the empty default must be filled by the caller.
    root : str | None
        Existing local dataset directory, or ``None`` for the Hub cache.
    revision : str | None
        Dataset revision used when fetching from the Hub.
    action_horizon : int
    video_backend : str | None
    base_image_key, wrist_image_key : str | None
        Dataset camera fields; None explicitly disables that view.
    recorded_in_degrees : bool
        Recorded first five joints are degrees; convert samples and statistics to radians.
        False preserves legacy normalized joint positions. Gripper percentages become [0, 1].

    Returns
    -------
    SFTDatasetSpec
        Samples contain six absolute joint targets and the configured RGB views.

    Raises
    ------
    AssertionError
        If the horizon, local root, or dataset metadata violates the SO101 contract.
    """
    assert isinstance(repo_id, str) and repo_id.strip(), "set SO101 repo_id explicitly"
    assert action_horizon >= 1, "action_horizon must be positive"
    assert type(recorded_in_degrees) is bool, "recorded_in_degrees must be boolean"
    dataset_root = Path(root) if root is not None else None
    assert dataset_root is None or dataset_root.is_dir(), (
        f"SO101 dataset root does not exist: {dataset_root}"
    )
    metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root, revision=revision)
    assert metadata.robot_type in ("so101_follower", "so_follower"), (
        f"expected so101_follower or so_follower, got {metadata.robot_type!r}"
    )
    assert metadata.fps >= 1, "SO101 dataset FPS must be positive"
    for key in ("observation.state", "action"):
        assert tuple(metadata.features[key]["shape"]) == (6,), f"SO101 {key} must have shape (6,)"
    image_keys = tuple(key for key in (base_image_key, wrist_image_key) if key is not None)
    assert image_keys, "SO101 requires at least one camera"
    for key in image_keys:
        feature = metadata.features[key]
        assert feature["dtype"] in ("video", "image") and feature["shape"][-1] == 3, (
            f"SO101 {key} must be RGB video or image"
        )

    kwargs: dict[str, Any] = {
        "root": dataset_root,
        "revision": revision,
        "delta_timestamps": {"action": [index / metadata.fps for index in range(action_horizon)]},
        "return_uint8": True,
    }
    if video_backend is not None:
        kwargs["video_backend"] = video_backend
    source = LeRobotDataset(repo_id, **kwargs)
    return SFTDatasetSpec(
        dataset=SO101SFTDataset(
            source, base_image_key=base_image_key, wrist_image_key=wrist_image_key,
            recorded_in_degrees=recorded_in_degrees,
        ),
        collate_fn=None,
        state_stats=model_stats(
            metadata.stats["observation.state"], use_degrees=recorded_in_degrees,
        ),
        action_stats=model_stats(metadata.stats["action"], use_degrees=recorded_in_degrees),
        image_keys=image_keys,
        state_key="observation/state",
        action_key="actions",
        task_key="prompt",
        embodiment="so101",
        joint_units="radians" if recorded_in_degrees else "normalized",
    )

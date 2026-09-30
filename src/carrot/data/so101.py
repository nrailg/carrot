"""SO101 LeRobot samples mapped to the PI0.5 transform contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from torch.utils.data import Dataset

from .dataset_spec import SFTDatasetSpec


class SO101SFTDataset(Dataset[dict[str, Any]]):
    """Expose source targets and explicit camera views.

    Parameters
    ----------
    source : LeRobotDataset
    base_image_key, wrist_image_key : str | None
        Omit a view with None; at least one camera must be explicitly configured.
    """

    def __init__(
        self, source: LeRobotDataset, *,
        base_image_key: str | None = None,
        wrist_image_key: str | None = None,
    ) -> None:
        self.source = source
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
        state = np.asarray(state, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
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
        ),
        collate_fn=None,
        state_stats=metadata.stats["observation.state"],
        action_stats=metadata.stats["action"],
        image_keys=image_keys,
        state_key="observation/state",
        action_key="actions",
        task_key="prompt",
        embodiment="so101",
    )

"""SO101 LeRobot samples mapped to the PI0.5 transform contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from torch.utils.data import Dataset

from .dataset_spec import SFTDatasetSpec


class SO101SFTDataset(Dataset[dict[str, Any]]):
    """Expose two camera views and absolute joint targets for PI0.5."""

    def __init__(self, source: LeRobotDataset) -> None:
        self.source = source

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.source[index]
        return {
            "observation/state": sample["observation.state"],
            "observation/image": sample["observation.images.top"],
            "observation/wrist_image": sample["observation.images.fpv"],
            "prompt": sample["task"],
            "actions": sample["action"],
            "action_is_pad": sample["action_is_pad"],
        }


def build_dataset(
    *,
    repo_id: str = "felixmayor/orange_cube_merged",
    root: str | None = None,
    revision: str | None = None,
    action_horizon: int = 50,
    video_backend: str | None = None,
) -> SFTDatasetSpec:
    """Load SO101 LeRobot v3 data with episode-bounded future actions.

    Parameters
    ----------
    repo_id : str
    root : str | None
        Existing local dataset directory, or ``None`` for the Hub cache.
    revision : str | None
        Dataset revision used when fetching from the Hub.
    action_horizon : int
    video_backend : str | None

    Returns
    -------
    SFTDatasetSpec
        Samples contain six absolute joint targets and two RGB views.

    Raises
    ------
    ValueError
        If the dataset metadata does not describe an SO101 follower.
    """
    if action_horizon < 1:
        raise ValueError("action_horizon must be positive")
    dataset_root = Path(root) if root is not None else None
    if dataset_root is not None and not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root, revision=revision)
    if metadata.robot_type != "so101_follower":
        raise ValueError(
            f"SO101 dataset requires robot_type='so101_follower', got {metadata.robot_type!r}"
        )
    if metadata.fps < 1:
        raise ValueError("SO101 dataset FPS must be positive")
    for key in ("observation.state", "action"):
        if tuple(metadata.features[key]["shape"]) != (6,):
            raise ValueError(f"SO101 {key} must have shape (6,)")
    for key in ("observation.images.top", "observation.images.fpv"):
        feature = metadata.features[key]
        if feature["dtype"] not in ("video", "image") or feature["shape"][-1] != 3:
            raise ValueError(f"SO101 {key} must be RGB video or image")

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
        dataset=SO101SFTDataset(source),
        collate_fn=None,
        state_stats=metadata.stats["observation.state"],
        action_stats=metadata.stats["action"],
        image_keys=("observation.images.top", "observation.images.fpv"),
        state_key="observation/state",
        action_key="actions",
        task_key="prompt",
        embodiment="so101",
    )

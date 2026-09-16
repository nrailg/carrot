"""Default LeRobot dataset integration and RobotWin preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.collate import lerobot_collate_fn

from .dataset_spec import SFTDatasetSpec
from .robotwin import robotwin_preprocess as robotwin_preprocess


def build_dataset(
    *,
    repo_id: str = "lerobot/robotwin_unified",
    root: str | None = None,
    video_backend: str | None = None,
    action_horizon: int = 50,
    image_keys: tuple[str, str, str] = (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ),
    state_key: str = "observation.state",
    action_key: str = "action",
    task_key: str = "task",
) -> SFTDatasetSpec:
    """Build the default LeRobot SFT dataset.

    Parameters
    ----------
    repo_id : str
    root : str | None
    video_backend : str | None
    action_horizon : int
    image_keys : tuple[str, str, str]
    state_key : str
    action_key : str
    task_key : str

    Returns
    -------
    SFTDatasetSpec
    """
    dataset_root = Path(root) if root else None
    metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    delta_timestamps = {action_key: [index / metadata.fps for index in range(action_horizon)]}
    kwargs: dict[str, Any] = {"root": dataset_root, "delta_timestamps": delta_timestamps}
    if video_backend is not None:
        kwargs["video_backend"] = video_backend
    dataset = LeRobotDataset(repo_id, **kwargs)
    return SFTDatasetSpec(
        dataset=dataset,
        collate_fn=lerobot_collate_fn if metadata.has_language_columns else None,
        state_stats=metadata.stats[state_key],
        action_stats=metadata.stats[action_key],
        image_keys=tuple(image_keys),
        state_key=state_key,
        action_key=action_key,
        task_key=task_key,
    )

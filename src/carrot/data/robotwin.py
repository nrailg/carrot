"""RoboTwin LeRobot dataset integration and joint and gripper conversions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.collate import lerobot_collate_fn

from .dataset_spec import SFTDatasetSpec


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


def robotwin_preprocess(
    state: torch.Tensor | np.ndarray, actions: torch.Tensor | np.ndarray | None
) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray | None]:
    """Convert RobotWin Aloha labels into PI0.5 SFT targets.

    Parameters
    ----------
    state : torch.Tensor | numpy.ndarray
    actions : torch.Tensor | numpy.ndarray | None
        Pass None for observation-only inference.

    Returns
    -------
    tuple[torch.Tensor | numpy.ndarray, torch.Tensor | numpy.ndarray | None]
    """
    if isinstance(state, torch.Tensor):
        flip = state.new_tensor([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])
        state = state.clone()
        state[..., :14] *= flip
        linear = 0.01844 + state[..., [6, 13]] * (0.05800 - 0.01844)
        ratio = (0.022**2 + linear**2 - 0.036**2) / (2 * 0.022 * linear)
        state[..., [6, 13]] = (torch.asin(torch.clamp(ratio, -1, 1)) - 0.5476) / (1.6296 - 0.5476)
    else:
        flip = np.asarray([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1], dtype=np.float32)
        state = state.copy()
        state[..., :14] *= flip
        linear = 0.01844 + state[..., [6, 13]] * (0.05800 - 0.01844)
        ratio = (0.022**2 + linear**2 - 0.036**2) / (2 * 0.022 * linear)
        state[..., [6, 13]] = (np.arcsin(np.clip(ratio, -1, 1)) - 0.5476) / (1.6296 - 0.5476)
    if actions is None:
        return state, None
    actions = actions.clone() if isinstance(actions, torch.Tensor) else actions.copy()
    actions[..., :14] *= flip
    actions[..., [6, 13]] = (-0.6213 + actions[..., [6, 13]] * (1.4910 + 0.6213)) - 0.5476
    delta_mask = [True] * 6 + [False] + [True] * 6 + [False]
    actions[..., delta_mask] -= state[..., delta_mask][..., None, :]
    return state, actions

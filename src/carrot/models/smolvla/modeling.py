"""Construct SmolVLA from LeRobot's model and dataset definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import nn


@dataclass(frozen=True)
class SmolVLAComponents:
    policy: nn.Module
    dataset: Any
    preprocessor: Any
    postprocessor: Any
    collate_fn: Any


def build_smolvla(
    *,
    model_path: str,
    dataset_repo_id: str,
    dataset_root: str | None,
    device: str,
    video_backend: str | None = None,
) -> SmolVLAComponents:
    """Build a fine-tuning policy and data pipeline with LeRobot factories."""
    try:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
        from lerobot.datasets.factory import resolve_delta_timestamps
        from lerobot.policies import make_policy, make_pre_post_processors
        from lerobot.utils.collate import lerobot_collate_fn
    except ImportError as error:
        raise ImportError('SmolVLA training requires pip install -e ".[sft]"') from error

    root = Path(dataset_root) if dataset_root is not None else None
    metadata = LeRobotDatasetMetadata(dataset_repo_id, root=root)
    policy_config = PreTrainedConfig.from_pretrained(model_path)
    policy_config.pretrained_path = model_path
    policy_config.device = device
    if policy_config.type != "smolvla":
        raise ValueError(f"expected a SmolVLA checkpoint, got policy type {policy_config.type!r}")

    policy = make_policy(cfg=policy_config, ds_meta=metadata)
    delta_timestamps = resolve_delta_timestamps(policy_config, metadata, {})
    dataset_kwargs: dict[str, Any] = {
        "root": root,
        "delta_timestamps": delta_timestamps,
    }
    if video_backend is not None:
        dataset_kwargs["video_backend"] = video_backend
    dataset = LeRobotDataset(dataset_repo_id, **dataset_kwargs)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        dataset_stats=metadata.stats,
        dataset_meta=metadata,
    )
    return SmolVLAComponents(
        policy=policy,
        dataset=dataset,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        collate_fn=lerobot_collate_fn,
    )

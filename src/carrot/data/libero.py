"""LIBERO LeRobot samples mapped to the PI0.5 inference request contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from torch.utils.data import Dataset

from .dataset_spec import SFTDatasetSpec


class LiberoSFTDataset(Dataset[dict[str, Any]]):
    """Expose raw LIBERO samples with the same fields as inference requests."""

    def __init__(self, source: LeRobotDataset) -> None:
        self.source = source

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.source[index]
        return {
            "observation/state": sample["observation.state"],
            "observation/image": sample["observation.images.image"],
            "observation/wrist_image": sample["observation.images.image2"],
            "prompt": sample["task"],
            "actions": sample["action"],
            "action_is_pad": sample["action_is_pad"],
        }


def build_dataset(
    *,
    root: str,
    repo_id: str = "lerobot/libero",
    action_horizon: int = 10,
    video_backend: str | None = None,
) -> SFTDatasetSpec:
    """Load local 10 FPS LIBERO data with ten-frame action chunks.

    Parameters
    ----------
    root : str
        Local LeRobot dataset directory; remote discovery is not performed here.
    repo_id : str
    action_horizon : int
        Must match the official PI0.5 LIBERO checkpoint horizon of ten.
    video_backend : str | None

    Returns
    -------
    SFTDatasetSpec
        Raw samples retain the inference field names and action padding mask.
    """
    if action_horizon != 10:
        raise ValueError("PI0.5 LIBERO requires action_horizon=10")
    dataset_root = Path(root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    if metadata.fps != 10:
        raise ValueError(f"LIBERO dataset must use 10 FPS, got {metadata.fps}")
    kwargs: dict[str, Any] = {
        "root": dataset_root,
        "delta_timestamps": {"action": [index / metadata.fps for index in range(10)]},
    }
    if video_backend is not None:
        kwargs["video_backend"] = video_backend
    source = LeRobotDataset(repo_id, **kwargs)
    return SFTDatasetSpec(
        dataset=LiberoSFTDataset(source),
        collate_fn=None,
        state_stats=metadata.stats["observation.state"],
        action_stats=metadata.stats["action"],
        image_keys=("observation.images.image", "observation.images.image2"),
        state_key="observation/state",
        action_key="actions",
        task_key="prompt",
        embodiment="libero",
    )

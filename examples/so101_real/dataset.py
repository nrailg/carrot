import os
from pathlib import Path

import datasets.config
import huggingface_hub.constants
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

from .config import JOINT_NAMES, DeploymentConfig
from .observations import DatasetSource


def load_dataset_source(config: DeploymentConfig, horizon: int) -> DatasetSource:
    """Open a local LeRobot episode using the SO101 training field mapping.

    Parameters
    ----------
    config : DeploymentConfig
        Local files must already correspond to dataset_revision; offline loading does
        not verify their Hub revision. Hub access is disabled for this process.
    horizon : int
        Future action window used only for comparison and episode-boundary checks.

    Returns
    -------
    DatasetSource
        Unnormalized observations and a separate demonstration action reference.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    huggingface_hub.constants.HF_HUB_OFFLINE = True
    datasets.config.HF_DATASETS_OFFLINE = True
    root = Path(config.dataset_root)
    for name in ("info.json", "stats.json", "tasks.parquet"):
        assert (root / "meta" / name).is_file(), f"missing local dataset meta/{name}"
    meta = LeRobotDatasetMetadata(config.dataset_repo, root=root, revision=config.dataset_revision)
    assert meta.robot_type == "so101_follower" and meta.fps == config.fps
    for name in ("observation.state", "action"):
        assert tuple(meta.features[name]["shape"]) == (6,)
        assert tuple(meta.features[name]["names"]) == JOINT_NAMES, f"wrong joint order: {name}"
    for name in ("observation.images.top", "observation.images.fpv"):
        assert meta.features[name]["dtype"] in ("image", "video")
        assert meta.features[name]["shape"][-1] == 3
    dataset = LeRobotDataset(
        config.dataset_repo, root=root, revision=config.dataset_revision,
        episodes=[config.episode], return_uint8=True,
        delta_timestamps={"action": [step / config.fps for step in range(horizon)]},
    )
    return DatasetSource(dataset, config.episode, config.start_frame, horizon, config.prompt)

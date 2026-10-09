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
        Complete local files; Hub access is disabled for this process.
    horizon : int
        Future action window used only for comparison and episode-boundary checks.

    Returns
    -------
    DatasetSource
        Source observations and a separate demonstration action reference.
    """
    # 两个库会缓存离线配置，仅设置环境变量不足以约束当前进程。
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    huggingface_hub.constants.HF_HUB_OFFLINE = True
    datasets.config.HF_DATASETS_OFFLINE = True
    root = Path(config.dataset_root)
    for name in ("info.json", "stats.json", "tasks.parquet"):
        assert (root / "meta" / name).is_file(), f"missing local dataset meta/{name}"
    meta = LeRobotDatasetMetadata(config.dataset_repo, root=root)
    assert meta.robot_type in ("so101_follower", "so_follower"), "expected a SO101 dataset"
    assert meta.fps == config.fps, "dataset FPS must match deployment"
    for name in ("observation.state", "action"):
        assert tuple(meta.features[name]["shape"]) == (6,)
        assert tuple(meta.features[name]["names"]) == JOINT_NAMES, f"wrong joint order: {name}"
    for camera in (() if config.frame_index_prompt_frames is not None
                   else config.image_keys.values()):
        name = f"observation.images.{camera}"
        assert name in meta.features, f"missing configured camera: {name}"
        assert meta.features[name]["dtype"] in ("image", "video")
        assert meta.features[name]["shape"][-1] == 3

    # 未来 action 窗口只供参考比较；其 padding 必须与 episode 边界一致。
    dataset = LeRobotDataset(
        config.dataset_repo, root=root,
        episodes=[config.episode], return_uint8=True,
        delta_timestamps={"action": [step / config.fps for step in range(horizon)]},
    )
    return DatasetSource(dataset, config, horizon)

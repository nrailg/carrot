import os
from pathlib import Path

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.pi05.configuration_pi05 import PI05Config

REPO_ID = "nrailg/arm_dance_20261008_225311_20261008_225313"
VENV = Path("/opt/venvs/lerobot-official-arm-dance")
WHEEL_SHA256 = "1894516040c65f80a45bd9741f8174aae90ed5d93da0627ab4f1a85fd8d75e90"


def resources() -> tuple[Path, Path, Path]:
    dfs = Path(os.environ["MY_DFS"])
    return (
        dfs / "hf-hub" / REPO_ID,
        dfs / "hf-hub/lerobot/pi05_base",
        dfs / "hf-hub/google/paligemma-3b-pt-224",
    )


def dataset_config(policy: PI05Config) -> TrainPipelineConfig:
    root, _, _ = resources()
    return TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=REPO_ID, root=str(root), video_backend="pyav"),
        policy=policy,
    )


def load_dataset(policy: PI05Config):
    return make_dataset(dataset_config(policy))

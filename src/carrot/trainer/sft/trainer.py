"""Controller for distributed supervised fine-tuning."""

from __future__ import annotations

import os

from carrot.distributed import Cluster, PlacementSpec, RolePlacement
from carrot.trainer.sft.config import SFTConfig
from carrot.trainer.sft.worker import SFTTrainWorker


def _worker_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for name in (
        "PYTHONPATH",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "WANDB_BASE_URL",
        "WANDB_API_KEY",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
    ):
        value = os.environ.get(name)
        if value:
            env[name] = value
    return env


class SFTTrainer:
    """Driver-side controller that owns cluster and train-worker lifecycles."""

    def __init__(self, config: SFTConfig, *, resume: str | None = None) -> None:
        self.config = config
        self.resume = resume

    def run(self) -> list[dict[str, float | int]]:
        cpus_per_worker = max(1, self.config.dataset.num_workers + 1)
        env_vars = _worker_env()
        with Cluster(address=os.environ.get("RAY_ADDRESS"), env_vars=env_vars) as cluster:
            cluster.reserve(
                "sft",
                PlacementSpec(
                    num_nodes=self.config.num_nodes,
                    bundles_per_node=self.config.gpus_per_node,
                    cpus_per_bundle=cpus_per_worker,
                    gpus_per_bundle=1,
                ),
            )
            workers = cluster.launch(
                "sft-train-worker",
                SFTTrainWorker,
                self.config,
                self.resume,
                placement=RolePlacement(
                    pool="sft",
                    cpus_per_actor=cpus_per_worker,
                    gpus_per_actor=1,
                ),
                env_vars=env_vars,
            )
            results = workers.call("train").wait()
            print(f"SFT finished: {results[0]}", flush=True)
            return results

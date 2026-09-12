"""Controller-side entry point for distributed SFT."""

from __future__ import annotations

import os

from carrot.distributed import Cluster, PlacementSpec, RolePlacement
from carrot.trainer.sft.config import SFTConfig
from carrot.trainer.sft.trainer import SFTTrainerWorker


def _worker_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for name in (
        "PYTHONPATH",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "SMOLVLM_PATH",
        "WANDB_BASE_URL",
        "WANDB_API_KEY",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
    ):
        value = os.environ.get(name)
        if value:
            env[name] = value
    return env


def run_sft(
    config: SFTConfig,
    *,
    resume: str | None = None,
) -> list[dict[str, float | int]]:
    cpus_per_worker = max(1, config.dataset.num_workers + 1)
    env_vars = _worker_env()
    with Cluster(address=os.environ.get("RAY_ADDRESS"), env_vars=env_vars) as cluster:
        cluster.reserve(
            "sft",
            PlacementSpec(
                num_nodes=config.num_nodes,
                bundles_per_node=config.gpus_per_node,
                cpus_per_bundle=cpus_per_worker,
                gpus_per_bundle=1,
            ),
        )
        workers = cluster.launch(
            "sft-trainer",
            SFTTrainerWorker,
            config,
            resume,
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

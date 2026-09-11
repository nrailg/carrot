"""Controller-side entry point for distributed SFT."""

from __future__ import annotations

from carrot.distributed import Cluster, PlacementSpec, RolePlacement
from carrot.trainer.sft.config import SFTConfig
from carrot.trainer.sft.trainer import SFTTrainerWorker


def run_sft(
    config: SFTConfig,
    *,
    resume: str | None = None,
) -> list[dict[str, float | int]]:
    cpus_per_worker = max(1, config.dataset.num_workers + 1)
    with Cluster() as cluster:
        cluster.reserve(
            "sft",
            PlacementSpec(
                bundles_per_node=config.num_gpus,
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
        )
        return workers.call("train").wait()

"""Launch a fit case with the standard Carrot distributed worker lifecycle."""

import argparse
import os
from pathlib import Path

import yaml

from carrot.distributed import Cluster, PlacementSpec, RolePlacement
from carrot.trainer.sft.config import SFTConfig
from carrot.trainer.sft.trainer import _worker_env
from recipes.pi05_sft_so101_fit_validation.fit import FitTrainWorker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    values = yaml.safe_load(args.config.read_text())
    fit = values.pop("fit")
    config = SFTConfig.from_dict(values)
    assert not Path(config.output_dir).exists(), "use a fresh output directory"
    cpus = max(1, config.dataset.num_workers + 1)
    env = _worker_env()
    with Cluster(address=os.environ.get("RAY_ADDRESS"), env_vars=env) as cluster:
        cluster.reserve("sft", PlacementSpec(num_nodes=config.num_nodes,
                        bundles_per_node=config.gpus_per_node,
                        cpus_per_bundle=cpus, gpus_per_bundle=1))
        workers = cluster.launch("fit-worker", FitTrainWorker, config,
                                 fit["vision"], fit["noise_seed"],
                                 placement=RolePlacement(pool="sft", cpus_per_actor=cpus,
                                                         gpus_per_actor=1), env_vars=env)
        results = workers.call("train").wait()
        print("SFT finished:", results[0], flush=True)


if __name__ == "__main__":
    main()

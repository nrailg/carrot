# Carrot

Carrot is a lightweight, backend-neutral foundation for distributed robot SFT
and RL training. The first milestone provides the orchestration primitives;
model, environment, and algorithm integrations come next.

## Design

The public API is intentionally independent of Ray:

```text
training runner
  ├── Cluster          resource view + lifecycle
  ├── PlacementSpec    worker count + per-worker resources
  ├── WorkerGroup      broadcast / scatter / rank selection
  ├── Future           asynchronous control plane
  └── Channel          bounded producer-consumer data plane
          │
          └── private Ray runtime
```

This keeps useful ideas from the reference projects without inheriting their
runtime coupling:

- slime: a small, explicit rollout/train loop and asynchronous handles;
- verl: separate resource, dispatch, execute, and collect concerns;
- RLinf: WorkerGroup, placement, and channels for embodied pipelines;
- G-Core/YATT: predictable topology and a path toward parallel controllers.

See [docs/architecture.md](docs/architecture.md) for the design boundaries.

## Quick start

```bash
python -m pip install -e ".[dev]"
pytest
python examples/distributed_smoke.py
```

```python
from carrot.distributed import Cluster, PlacementSpec, RolePlacement, Worker


class Trainer(Worker):
    def step(self, batch):
        return {"rank": self.rank, "size": len(batch)}


if __name__ == "__main__":
    with Cluster() as cluster:
        cluster.reserve("train", PlacementSpec(bundles_per_node=2))
        trainers = cluster.launch(
            "trainer",
            Trainer,
            placement=RolePlacement(pool="train"),
        )
        print(trainers.call("step", [1, 2, 3]).wait())
```

Ray is a required implementation dependency, but Ray actors, object references,
queues, and scheduling types are kept behind Carrot's public API.

## SmolVLA SFT

Install the optional training dependencies and launch the bundled RoboTwin
configuration:

```bash
python -m pip install -e ".[sft]"
carrot-train-sft --config configs/smolvla_robotwin_sft.yaml
```

The `SFTTrainer` controller launches one `SFTTrainWorker` per GPU. Each worker uses
LeRobot 0.6.1's SmolVLA model, dataset, preprocessing, and flow-matching
objective — the same version as `wepsdl/carrot:v1.0` — and owns its local
FSDP2-wrapped model, optimization loop, and distributed checkpoint participation.
Set `--resume` to a `checkpoints/step-*` directory to restore model, optimizer,
scheduler, and step.

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

## PI0.5 SFT

Install the optional training dependencies and launch the bundled RoboTwin
configuration:

```bash
python -m pip install -e ".[sft]"
carrot-train-sft --config configs/pi05_robotwin_sft.yaml
```

Carrot vendors the native PI0.5 architecture derived from verl-vla: a SigLIP
vision encoder plus dual-stream PaliGemma/action-expert decoder. It uses FSDP2
with BF16 forward parameters and FP32 optimizer masters. RobotWin data adaptation
is intentionally being rebuilt for PI0.5; no legacy-policy compatibility code remains.

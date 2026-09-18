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

Generate the PI0.5 statistics once before the first training run:

```bash
carrot-compute-pi05-stats \
  --root /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/lerobot/robotwin_unified \
  --output /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/lerobot/robotwin_unified/meta/pi05_norm_stats.json
```

The statistics command streams only parquet columns. It does not decode camera
video. Action statistics are computed from the same 50-step, episode-bounded
future action windows used for training, after Aloha adaptation and conversion
of arm joints to deltas relative to the current state.

Carrot vendors the native PI0.5 architecture derived from verl-vla: a SigLIP
vision encoder plus dual-stream PaliGemma/action-expert decoder. It uses FSDP2
with BF16 forward parameters and FP32 optimizer masters. RobotWin data adaptation
constructs a 50-step action window, maps the three camera streams to 224x224
PI0.5 inputs, encodes the task and discretized state into a 200-token prompt,
and excludes padded action steps from the flow-matching loss.

The LeRobot dataset must provide these fields per frame:

- `observation.state`: 14 Aloha state values;
- `action`: 14 absolute Aloha action values (the dataset adapter builds the
  `[50, 14]` future window);
- the three image keys returned by `dataset.factory`, as CHW tensors in
  `[0, 1]` or `[0, 255]`;
- `task`: the language instruction. LeRobot also supplies `action_is_pad` for
  future steps beyond the end of an episode.

`dataset.factory` is an import path returning `carrot.data.SFTDatasetSpec`; its
`factory_kwargs` are passed through unchanged. `dataset.preprocess` is an
optional import path for a function taking `(state, actions)` and returning the
transformed pair. The RobotWin defaults are
`carrot.data.lerobot.build_dataset` and
`carrot.data.lerobot.robotwin_preprocess`; set `preprocess: null` when no
dataset-specific state/action conversion is required.

Each `checkpoints/step-*` directory contains two representations:

- `dcp/` is the sharded model and AdamW state used by `--resume`;
- `pretrained_model/` is a consolidated Diffusers checkpoint, tokenizer, and
  `norm_stats.json` loadable with `PI0Pytorch.from_pretrained`.

RoboTwin's LeRobot metadata does not contain PI0.5 quantiles. Set
`dataset.norm_stats_path` to an OpenPI-style JSON containing `state.q01/q99`
and `action.q01/q99`; when omitted, the adapter uses dataset min/max statistics.

Resume a distributed checkpoint with:

```bash
carrot-train-sft --config configs/pi05_robotwin_sft.yaml \
  --resume outputs/pi05_robotwin_sft/checkpoints/step-00001000
```

This restores model parameters, AdamW state, scheduler state, and the completed
step from `dcp/`. To start a fresh optimizer from an exported model instead,
point both `model.path` and `model.tokenizer_path` at that checkpoint's
`pretrained_model/` directory and omit `--resume`; the bundled
`norm_stats.json` is loaded automatically when `dataset.norm_stats_path` is
unset.

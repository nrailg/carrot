# Carrot

Carrot is a lightweight, backend-neutral foundation for distributed robot learning.
It provides resource placement, worker groups, and communication channels, with
PI0.5 SFT, checkpoint export and resume, LIBERO evaluation, and SO101 real-robot
inference built on top.

## Design

The public API keeps Ray actors, object references, queues, and placement groups
inside the private runtime:

```text
training / models / environments
              ↓
Cluster · WorkerGroup · Channel · PlacementSpec
              ↓
       private Ray runtime
```

See [architecture](docs/architecture.md) for the interfaces and module boundaries.

## Quickstart

Use Python **3.12**. For a standalone CPU runtime demo, create a local environment
and install the core package and development tools:

```bash
uv venv --python 3.12 .venv
uv --no-config pip install --python .venv/bin/python -e ".[dev]"
source .venv/bin/activate
python examples/distributed_smoke.py
```

The [demo](examples/distributed_smoke.py) sends trajectories from two environment
workers to a learner through a bounded channel. GPU containers and Mac robot
clients use separate environments; see [environment setup](docs/environments.md).

## Record SO101 demonstrations

Calibrate the leader and follower, check teleoperation and the wrist camera, then
record locally with LeRobot 0.6.1. In the recording environment, set the actual
ports, calibrated robot IDs, and camera index as described in the
[recording guide](docs/so101-data-sft.md):

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" --robot.use_degrees=true \
  --robot.cameras="{wrist: {type: opencv, index_or_path: $WRIST_CAMERA_INDEX, width: 640, height: 480, fps: 15, warmup_s: 5}}" \
  --teleop.type=so101_leader --teleop.port="$LEADER_PORT" \
  --teleop.id="$LEADER_ID" --teleop.use_degrees=true \
  --dataset.repo_id=local/knock_down_the_cylinder \
  --dataset.single_task="Knock down the cylinder" \
  --dataset.num_episodes=2 --dataset.episode_time_s=120 \
  --dataset.reset_time_s=60 --dataset.fps=15 \
  --dataset.push_to_hub=false --display_data=false
```

This records two episodes at 15 FPS without uploading them. Validate the videos,
joint units, and episode metadata before transferring the dataset for training.

## PI0.5 SFT

Start from a [SO101 wrist-camera configuration](recipes/pi05_sft_so101_knock_down_the_cylinder_overfit/train.yaml)
or the [RoboTwin configuration](configs/pi05_sft_robotwin_main.yaml).
Copy and adapt the model, tokenizer, dataset, statistics, output directory, and
GPU layout for the new run. In the prepared GPU environment, from the repository root:

```bash
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
python -m carrot.cli.train_sft --config /absolute/path/to/train.yaml
```

Training uses Ray workers and PyTorch FSDP2. See [SFT and checkpoints](docs/pi05-sft.md)
for data contracts, normalization, and resume, and [SO101 deployment](examples/so101_real/README.md)
for inference and robot execution. Training completion, offline fit, and task
success are evaluated separately.

## Documentation

| Document | Contents |
| --- | --- |
| [Environment setup](docs/environments.md) | Python 3.12, local `.venv`, Mac recording/client environments, and GPU containers |
| [SO101 recording to SFT](docs/so101-data-sft.md) | Recording, dataset checks, transfer, training preparation, and experiment handoff |
| [PI0.5 SFT and checkpoints](docs/pi05-sft.md) | Training entry points, data contracts, statistics, export, and resume |
| [SO101 deployment](examples/so101_real/README.md) | GPU policy service, Mac client, observations, robot execution, and logs |
| [Distributed architecture](docs/architecture.md) | Cluster, WorkerGroup, Channel, placement, and the Ray boundary |

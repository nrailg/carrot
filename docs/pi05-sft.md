# PI0.5 SFT and checkpoints

Activate the GPU environment and load the source using the
[environment guide](environments.md). See the [recipe index](../recipes/README.md)
for experiment records and [SO101 recording to SFT](so101-data-sft.md) for new data.

## Configure and launch

| Dataset / experiment | Configuration or record | Purpose |
| --- | --- | --- |
| RoboTwin ALOHA | [Training configuration](../configs/pi05_sft_robotwin_main.yaml) | 14-dimensional state/action, three cameras, horizon 50 |
| SO101 cylinder | [Training configuration](../recipes/pi05_sft_so101_knock_down_the_cylinder_overfit/train.yaml) | Six absolute joint targets, wrist camera, horizon 10 |
| SO101 Arm Dance | [Recipe](../recipes/pi05_sft_so101_arm_dance/README.md) | No-vision diagnostic with state jitter |
| Official LeRobot comparison | [Recipe](../recipes/lerobot_pi05_arm_dance/README.md) | Dedicated environment and runner, separate from Carrot training |
| LIBERO | [Recipe](../recipes/pi05_libero_sft_2k_20260919/README.md) | Panda task SFT and closed-loop evaluation |

Absolute paths and resource layouts in these configurations belong to historical
experiments. Copy a configuration for a new run and update the model, tokenizer,
dataset root, statistics, `output_dir`, training steps, save interval, and W&B
settings. Use a new output directory.

The dataset's `action_horizon` must match the loaded model configuration.
`global_batch_size` must be divisible by `micro_batch_size * dp_size`, and
`dp_size` must be divisible by `num_nodes`.

SFT allocates one Ray worker per GPU and trains with PyTorch FSDP2. For multiple
nodes, prepare a Ray cluster with the required resources and set `RAY_ADDRESS`.
Single-node runs also need sufficient GPUs and CPUs. From the repository root,
launch the configuration prepared for this run:

```bash
python -m carrot.cli.train_sft --config /absolute/path/to/train.yaml
```

Module entry points do not require an installed console script. Environments
where Carrot is explicitly installed can also use `carrot-train-sft`.

## Data and normalization

`dataset.factory` returns an [SFTDatasetSpec](../src/carrot/data/dataset_spec.py).
LeRobot supplies episode-bounded future action windows, task text, and
`action_is_pad`. Each embodiment adapter defines its camera, state/action, and
preprocessing contracts.

- **RoboTwin:** 14-dimensional ALOHA data uses
  [dedicated preprocessing](../src/carrot/data/robotwin.py) for joint signs,
  gripper conversion, and motion-joint action deltas relative to the current
  state. Do not apply this preprocessing to SO101.
- **SO101:** the [adapter](../src/carrot/data/so101.py) preserves six-dimensional
  source state/action values. Specify camera keys explicitly. A wrist-only
  dataset uses `base_image_key: null`,
  `wrist_image_key: observation.images.wrist`, and `dataset.preprocess: null`.

Select the normalization source explicitly:

| Configuration | Behavior |
| --- | --- |
| `norm_stats_source: file` | Read `norm_stats_path`, or the model directory's `norm_stats.json` when no path is set. Missing files fail |
| `norm_stats_source: dataset` | Use dataset metadata state/action statistics; `norm_stats_path` must be null |

Existing SO101 recipes use dataset q01/q99 statistics and export the effective
statistics with the checkpoint. Use the new dataset's own statistics rather
than reusing cylinder or Arm Dance statistics.

The RoboTwin configuration uses a separate statistics file. Set the actual
`ROBOTWIN_ROOT`, then compute statistics before training:

```bash
python -m carrot.cli.compute_pi05_stats \
  --root "${ROBOTWIN_ROOT:?set the RoboTwin dataset root}" \
  --output "$ROBOTWIN_ROOT/meta/pi05_norm_stats.json" \
  --action-horizon 50
```

This command defaults to RoboTwin preprocessing and streams parquet without
decoding videos. Set `dataset.norm_stats_path` to the output file. Match the
horizon and preprocessing to training. The file-based source does not fall back
to dataset min/max when the file is missing.

## Save and resume

Carrot writes each checkpoint directly to
`output_dir/checkpoints/step-XXXXXXXX/`:

```text
step-XXXXXXXX/
  config.json
  model.safetensors
  norm_stats.json
  tokenizer files
  train_config.yaml
  trainer_state.json
  optimizer/             # optimizer-only PyTorch DCP
```

The root is an OpenPI PyTorch model export; `optimizer/` contains optimizer DCP.
Configurations using `norm_stats_asset_id` also copy the specified statistics
file to `assets/<asset_id>/`. See the
[save implementation](../src/carrot/trainer/sft/checkpoint.py).

```bash
python -m carrot.cli.train_sft \
  --config /absolute/path/to/train.yaml \
  --resume /absolute/path/to/checkpoints/step-00001000
```

Resume loads the model from the checkpoint root and restores optimizer,
scheduler, and step. The configuration still selects tokenizer and statistics;
keep them consistent with the original run.

Model parameters resume at the export precision. A BF16 export does not preserve
full FP32 master precision, and complete RNG state restoration is not implemented.
See the [checkpoint test record](../tests/pi_05/test_pi05_checkpoint_distributed.md)
for the validation limits of real PI0.5 two-GPU optimizer-only resume.

To start a new optimizer from an existing export, point `model.path` and
`model.tokenizer_path` at the checkpoint root, omit `--resume`, and use a new
output directory. To use the checkpoint's statistics, set
`norm_stats_source: file` and `norm_stats_path: null`.

Serve a SO101 checkpoint with:

```bash
python -m carrot.cli.serve_pi05_policy \
  --embodiment so101 \
  --checkpoint /absolute/path/to/checkpoints/step-00002000 \
  --device cuda:0 --port 8080
```

Starting the service does not control the robot. See
[SO101 deployment](../examples/so101_real/README.md) for client checks, live
cameras, and execution protocols. Validate training completion, offline fit,
and real task success separately.

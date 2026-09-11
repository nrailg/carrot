# SmolVLA SFT progress

Last updated: 2026-09-12

## Completed

- Downloaded `lerobot/smolvla_base` to
  `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/lerobot/smolvla_base`.
- Downloaded `lerobot/robotwin_unified` to
  `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/lerobot/robotwin_unified`.
- Confirmed the dataset has state/action data and three visual observations.
- Added a `carrot.trainer.sft` package, CLI, YAML configuration, Ray worker
  orchestration, gradient accumulation, logging, and final/periodic checkpoints.
- Reused LeRobot 0.4.4 for the SmolVLA policy, dataset, processor, collator, and
  flow-matching training loss.
- Added composable PyTorch FSDP2 modeling. SmolVLA uses an explicit hook-safe
  sharding plan because the upstream policy calls parts of the flow model
  directly and does not provide one linear module execution order.
- Matched the optimizer and cosine warmup/decay defaults from the LeRobot
  SmolVLA preset.
- Added distributed checkpoint save/resume for model, optimizer, scheduler,
  and global step.

## Validation

- `ruff check src tests`: passed.
- SFT/modeling/checkpoint unit tests: 12 passed.
- The full suite reached 16 passing tests, but seven pre-existing Ray runtime
  tests could not start GCS in the restricted local environment because local
  IP/hostname discovery failed.

## Pending after code approval

- Sync this worktree to the Gemini CephFS code location.
- Run a small multi-GPU smoke test against the downloaded checkpoint and
  dataset, then tune batch size/worker count from observed GPU memory and data
  throughput.

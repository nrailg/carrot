# SFT trainer smoke

- Python: `test_sft_trainer.py` checks scheduler floor, failed-peer teardown, and an opt-in two-step PI0.5 Ray GPU smoke.
- Run: set `MY_DFS`, `CARROT_PI05_MODEL_PATH`, `CARROT_PI05_TOKENIZER_PATH`, and
  `CARROT_ROBOTWIN_ROOT` to existing assets;
  start Ray according to `carrot-test`, then `bash tests/sft/test_sft_trainer.sh`.
- The runner checks assets before testing; the Python smoke no longer embeds a historical CephFS path.
- Missing model, tokenizer, or dataset variables now fail the two-step GPU pytest case.
- Separate CLI diagnostics use `pi05_sft_robotwin_smoke.yaml` (one step) and
  `pi05_sft_robotwin_preflight.yaml` (20 steps) in this directory. The Python
  two-step smoke above constructs its config in code and does not load either YAML.
- Historical image tag, Carrot commit, and separate result: not recorded.
- Status after relocation: `NOT RUN`.
- Carrot commit at relocation: `72d8168540d74e890358003cbbb4886f5c7ca0e4`.
- Docker image for next GPU run: `wepsdl/carrot:v1.10-driver-575.57.08-cuda-12.8` (planned, not historical).

## 2026-09-26 H20 run: PASS after test asset fix

The merged source commit was `eddfffbda2c2980bf1563266da8275093d137bc2`
on branch `nrwu/deployReal`. The remote Carrot checkout was synced to that tree;
this run also includes the test-only configuration changes on this branch.
The base PI0.5 checkpoint has no root `norm_stats.json` or tokenizer, so the
two-step test now explicitly selects dataset stats and takes a separate
`CARROT_PI05_TOKENIZER_PATH`. The runner checks `tokenizer.model` before pytest.

```bash
export MY_DFS=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu
export CARROT_PI05_MODEL_PATH="$MY_DFS/hf-hub/Physical-Intelligence/pi05_base_pytorch"
export CARROT_PI05_TOKENIZER_PATH="$MY_DFS/hf-hub/google/paligemma-3b-pt-224"
export CARROT_ROBOTWIN_ROOT="$MY_DFS/hf-hub/lerobot/robotwin_unified"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RAY_ADDRESS="${__HOST_IP__}:6379"
bash tests/sft/test_sft_trainer.sh
```

On `mpi-launcher@mpi-1759754893-launcher` with `/opt/venvs/carrot` and a
single-node Ray head started with source `PYTHONPATH`, the result was
`3 passed in 118.86s`, exit 0. Both optimizer steps completed; the final
result was `step=2`, `loss=0.13671875`. The earlier run failed before training
because the test pointed `tokenizer_path` at the model-only directory; that
diagnostic is retained in `sft_gpu_initial.log`. Passing output is in
`${MY_DFS}/experiments/carrot/test-runs/pi05-pr32-20260926T043636Z/sft_gpu_pass.log`.

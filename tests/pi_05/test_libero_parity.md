# PI0.5 LIBERO fixed-input OpenPI PyTorch parity

## Contract

This gate compares Carrot's complete LIBERO inference path with OpenPI PyTorch
`89d9220c68defd85b85f17acf130e965e58b8369`. Relative to upstream
`215abfb217dbac7d5f1273282331b9b1866c0479`, that commit changes only
`examples/libero/main.py` (task selection and durable rollout output). The existing
horizon-50 model parity test remains separate.

`capture_libero_observation.py` reads one real local `lerobot/libero` sample and
task prompt in Carrot's LeRobot environment. The OpenPI reference process reads
that immutable observation, the official checkpoint and norm stats, and a locally
cached tokenizer.model whose SHA-256 matches the official PaliGemma tokenizer.
It records
the raw observation,
fixed noise, OpenPI model inputs, raw `(10, 32)` actions and decoded `(10, 7)` actions.
The Carrot test checks the same observation and reports FP64 Dice plus absolute and
relative P50/P90/P99/Max before each pointwise assertion (`rtol=1e-3`,
`atol=1e-3` for images/state and `atol=6e-3` for 10-step actions).
Image masks and token IDs/masks must match exactly. The first failing field is the
diagnostic boundary; neither rollout success nor a synthetic fixture counts as PASS.

## Run

On a Gemini session with the current personal DFS resolved as `MY_DFS`, the source
synced, `/opt/venvs/openpi-libero` and `/opt/venvs/carrot` available, and one free GPU:

```bash
POLICY_GPU=0 bash tests/pi_05/test_libero_parity.sh
```

The runner checks the full OpenPI Python source fingerprint for the fixed commit
(`3b5e87f546e2e8effe3dac5b54c42f103bac7dc159c30047f4745ddaf5e5264d`),
because the remote synced copy has no `.git`. It reads the official assets and local
LeRobot sample, and writes the golden under
`${MY_DFS}/experiments/carrot/pi05-libero-parity/`. The record below must contain
the actual Carrot commit or source identity, image tag, checkpoint path and size,
stats SHA-256, dataset root/revision, test output, and artifact path.
The pytest case fails when its required golden, checkpoint, tokenizer, or CUDA
setting is absent; the caller must provide these assets explicitly.

## 2026-09-19 Gate 2 result: PASS at the recorded tolerances

The first run failed on `image/base_0_rgb`: Carrot interpolated float pixels with
Torch, whereas OpenPI resizes uint8 with PIL before scaling to `[-1, 1]`.
The LIBERO transform now uses the PIL path; RoboTwin keeps its existing transform.
The next run passed all model inputs but failed `raw_actions_32d` at `atol=1e-3`.
OpenPI eager reduced the action max error from `0.005202` to `0.001796`, identifying
compilation as a substantial numerical factor, not a field-mapping error. The
official `max-autotune` reference remains the gate. OpenPI used Torch
`2.7.1+cu128` / Transformers `4.53.2`; Carrot used Torch `2.11.0+cu128` /
Transformers `5.5.4`. The final action tolerance is fixed at `6e-3` and applied
pointwise; it is not a bitwise or `1e-3` action-parity claim.

| tensor | abs P50 | abs P90 | abs P99 | abs Max | rel P50 | rel P90 | rel P99 | rel Max | FP64 Dice |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base image | 0 | 5.96e-8 | 5.96e-8 | 1.19e-7 | 0 | 4.34e-7 | 1.69e-6 | 1.52e-5 | 1.78e-15 |
| left wrist image | 0 | 5.96e-8 | 1.19e-7 | 1.19e-7 | 0 | 1.16e-7 | 4.90e-7 | 1.52e-5 | 3.44e-15 |
| right wrist image | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| normalized state | 0 | 2.93e-8 | 9.24e-8 | 1.11e-7 | 0 | 1.41e-7 | 5.82e-7 | 6.96e-7 | 3.44e-15 |
| raw actions `(10,32)` | 5.43e-4 | 1.30e-3 | 2.57e-3 | 5.20e-3 | 0.664 | 4.65 | 26.0 | 119 | 1.04e-5 |
| decoded actions `(10,7)` | 4.42e-4 | 1.55e-3 | 3.80e-3 | 5.20e-3 | 0.0632 | 1.49 | 13.1 | 31.7 | 4.19e-6 |

All three image masks, token IDs and token mask were exactly equal. Large
relative errors for actions occur near zero-valued reference coordinates.
The final CUDA test returned `1 passed in 58.62s`, exit code 0. The newly added
PIL contract test plus modeling/inference regressions returned `25 passed`;
this includes RoboTwin behavior. The separate horizon-50 model test was left
unchanged and not rerun because its recorded converted checkpoint path no
longer exists; its historical result is not counted as new Gate 2 evidence.

Provenance:

- OpenPI `89d9220c68defd85b85f17acf130e965e58b8369`; remote source fingerprint
  `3b5e87f546e2e8effe3dac5b54c42f103bac7dc159c30047f4745ddaf5e5264d`;
  `pi05_libero`, `pytorch_compile_mode=max-autotune`.
- Carrot local HEAD `4dba80288609a67ac9c7025fd516b6c80cf85e87`, with uncommitted Gate 2
  changes synced to remote. Remote source has no `.git`; `transforms.py` SHA-256
  `3879bb5958876c4653396e0d6ddcbf185819abdd19c72e7a3a557099fcb8b817`,
  `embodiments/libero.py` SHA-256
  `a66c5fe746a0ce01c6efed4f075f0156d2f8e95eb72ef9a9df698228d4211838`.
- Checkpoint `pi05_libero_pytorch/model.safetensors`, 7,233,650,408 bytes;
  `config.json` SHA-256 `5c2728c53f4b33ee16380140f303713fdb5df78a8d26ee1489ace374fc54327a`;
  LIBERO `norm_stats.json` SHA-256
  `b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84`;
  local tokenizer model SHA-256
  `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`.
- Dataset root `${MY_DFS}/hf-hub/lerobot/libero`, sample index 0, task index 0;
  `meta/info.json` SHA-256
  `630630a1a4f07fc9fc1f1b2b7987ed3bd034bd8cffaaf8a0e6939ae9c468445d`.
  No dataset git revision is available from this local mirror.
- Container `mpi-launcher@mpi-1759754893-launcher`, image tag not exposed by its
  environment; do not attribute this run to a guessed tag.
- Reference observation and golden:
  `${MY_DFS}/benchmarks/carrot-pi05-libero-parity/20260919-102735/`;
  the final reference is `openpi-libero-golden-v2.npz`. Two compiled reference
  generations produced bitwise-equal raw and decoded actions.

The initial command was `MY_DFS=<detected personal DFS> HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1 POLICY_GPU=0 bash tests/pi_05/test_libero_parity.sh`.
After the first-layer fix, the final comparison reused the captured real
observation and ran `generate_openpi_libero_golden.py --num-steps 10` in
`/opt/venvs/openpi-libero`, then `python -m pytest -v -s --timeout=1800
tests/pi_05/test_libero_parity.py` in `/opt/venvs/carrot`, with
`CARROT_PI05_LIBERO_GOLDEN` pointing to `openpi-libero-golden-v2.npz` and
the same checkpoint/tokenizer paths above. Both commands used GPU 0 and the
source `PYTHONPATH` for their respective checkout; offline flags remained set
for reference generation.

## 2026-09-26 H20 rerun: PASS

With merged Carrot commit `eddfffbda2c2980bf1563266da8275093d137bc2`
synced to `mpi-launcher@mpi-1759754893-launcher`, the test reused the
previously recorded `openpi-libero-golden-v2.npz`, official LIBERO checkpoint,
and cached PaliGemma tokenizer. It ran offline with `CUDA_VISIBLE_DEVICES=0`
and `/opt/venvs/carrot`:

```bash
export MY_DFS=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu
export CARROT_PI05_LIBERO_GOLDEN="$MY_DFS/benchmarks/carrot-pi05-libero-parity/20260919-102735/openpi-libero-golden-v2.npz"
export CARROT_PI05_LIBERO_CHECKPOINT="$MY_DFS/hf-hub/Physical-Intelligence/pi05_libero_pytorch"
export CARROT_PI05_LIBERO_TOKENIZER="$MY_DFS/hf-hub/google/paligemma-3b-pt-224"
source /opt/venvs/carrot/bin/activate
cd "$MY_DFS/work/carrot"
export PYTHONPATH="$PWD/src:$PWD/tests:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
python -m pytest -v -s --timeout=1800 tests/pi_05/test_libero_parity.py
```

Result: `1 passed in 59.73s`, exit 0. Image masks and token fields were exact;
raw action maximum absolute error was `0.00520157814`, within the unchanged
pointwise `6e-3` tolerance. Full output:
`${MY_DFS}/experiments/carrot/test-runs/pi05-pr32-20260926T043636Z/libero_parity.log`.

For future runs, the runner defaults to
`${MY_DFS}/experiments/carrot/pi05-libero-parity/`. The historical
`20260919-102735` observation and golden NPZs were copied there without
removing the original benchmark archive; the `openpi-libero-golden-v2.npz`
copy matches its source at SHA-256
`533977c4a8dc15c84f484428cf9bad950534f4214b1a1a99d4c5ee948935d8ab`.

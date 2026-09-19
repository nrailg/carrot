# PI0.5 LIBERO fixed-input OpenPI PyTorch parity

## Contract

This gate compares Carrot's complete LIBERO inference path with OpenPI PyTorch
`89d9220c68defd85b85f17acf130e965e58b8369`. Relative to upstream
`215abfb217dbac7d5f1273282331b9b1866c0479`, that commit changes only
`examples/libero/main.py` (task selection and durable rollout output). The existing
horizon-50 model parity test remains separate.

`generate_openpi_libero_golden.py` reads one real local `lerobot/libero` sample,
task prompt, official checkpoint and norm stats. It records the raw observation,
fixed noise, OpenPI model inputs, raw `(10, 32)` actions and decoded `(10, 7)` actions.
The Carrot test checks the same observation and reports FP64 Dice plus absolute and
relative P50/P90/P99/Max before each pointwise assertion (`rtol=atol=1e-3`).
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
`${MY_DFS}/benchmarks/carrot-pi05-libero-parity/`. The record below must contain
the actual Carrot commit or source identity, image tag, checkpoint path and size,
stats SHA-256, dataset root/revision, test output, and artifact path.

## 2026-09-19 implementation status

- `NOT RUN`: real checkpoint and GPU validation has not yet completed.
- OpenPI reference revision: `89d9220c68defd85b85f17acf130e965e58b8369`.
- Carrot source identity, image, dataset revision, numerical rows and final verdict:
  pending execution. Do not infer PASS from local collection or earlier behavior benchmarks.

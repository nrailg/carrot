# PI0.5 fixed-input OpenPI parity

## Scope and provenance

- `test_openpi_parity.py` is the strict Carrot vs official OpenPI PyTorch one-step gate;
  JAX is diagnostic only. It does not cover LIBERO transforms or simulator behavior.
- Source checkout at reorganization: Carrot `72d8168540d74e890358003cbbb4886f5c7ca0e4`.
- Golden files require OpenPI commit `215abfb217dbac7d5f1273282331b9b1866c0479`.
- Docker image for the next run: `wepsdl/carrot:v1.10-driver-575.57.08-cuda-12.8`
  (build pending; no parity result is attributed to this image yet).
- The 2026-09-14 historical run did not record its Carrot commit or image tag. It failed
  the strict image-embedding assertion; the compact diagnostic summary below preserves the
  recorded distribution metrics.

## Execution

The parity test is opt-in because it loads the converted official PyTorch checkpoint and requires
a GPU. It treats official OpenPI PyTorch as the strict gate; JAX outputs are retained only for
diagnostics.
Missing golden/checkpoint variables or CUDA now fail the pytest case explicitly;
the opt-in runner checks that both golden files are nonempty before invoking it.

Generate deterministic JAX and PyTorch reference outputs from the same official OpenPI revision:
Set `ARTIFACT_DIR` below `${MY_DFS}/experiments`; do not put generated NPZs at
the DFS root. The generator scripts require explicit `--output` paths. Verify
that the OpenPI model source matches `215abfb217dbac7d5f1273282331b9b1866c0479`
before recording that commit; the synced Gemini copy has no `.git`.

```bash
: "${MY_DFS:?resolve the current personal DFS first}"
OPENPI_DIR="$MY_DFS/work/openpi"
CARROT_DIR="$MY_DFS/work/carrot"
OPENPI_COMMIT=215abfb217dbac7d5f1273282331b9b1866c0479
ARTIFACT_DIR="${MY_DFS}/experiments/carrot/pi05-openpi-parity/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$ARTIFACT_DIR"
source /opt/venvs/openpi-libero/bin/activate
cd "$OPENPI_DIR"
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python "$CARROT_DIR/tests/pi_05/generate_openpi_jax_golden.py" \
  --checkpoint "$MY_DFS/hf-hub/Physical-Intelligence/pi05_base" \
  --openpi-commit "$OPENPI_COMMIT" \
  --output "$ARTIFACT_DIR/pi05-openpi-jax-golden.npz"

CUDA_VISIBLE_DEVICES=0 python "$CARROT_DIR/tests/pi_05/generate_openpi_pytorch_golden.py" \
  --checkpoint "$MY_DFS/hf-hub/Physical-Intelligence/pi05_base_pytorch" \
  --jax-golden "$ARTIFACT_DIR/pi05-openpi-jax-golden.npz" \
  --openpi-commit "$OPENPI_COMMIT" \
  --output "$ARTIFACT_DIR/pi05-openpi-pytorch-golden.npz"
```

Run the Carrot parity gate:

```bash
source /opt/venvs/carrot/bin/activate
cd "$CARROT_DIR"
export PYTHONPATH="$PWD/src:$PWD/tests:${PYTHONPATH:-}"
export CARROT_PI05_OPENPI_GOLDEN="$ARTIFACT_DIR/pi05-openpi-jax-golden.npz"
export CARROT_PI05_OPENPI_PYTORCH_GOLDEN="$ARTIFACT_DIR/pi05-openpi-pytorch-golden.npz"
export CARROT_PI05_OPENPI_PYTORCH_CHECKPOINT="$MY_DFS/hf-hub/Physical-Intelligence/pi05_base_pytorch"
CUDA_VISIBLE_DEVICES=0 bash tests/pi_05/test_openpi_parity.sh
```

The test checks strict checkpoint loading, representative language/action/vision weights,
intermediate embeddings, the first denoising velocity, and one Euler sampling step. It does not
test tokenization, normalization, action unnormalization, or simulator behavior.

For every rerun, record the actual image tag, Carrot/OpenPI commits, golden and checkpoint
identities, FP64 Dice, P50/P90/P99/Max, pointwise assertion outcome, and artifact location here.

## 2026-09-19 local reorganization check

- `SKIP`: parity test collected but did not run because this devcloud shell has no configured
  golden files, checkpoint, or CUDA gate. No numerical parity conclusion follows.
- Carrot commit at check: `72d8168540d74e890358003cbbb4886f5c7ca0e4`;
  Docker image: not applicable (local devcloud).

## Migrated historical record

- The 2026-09-14 three-way run used OpenPI commit
  `215abfb217dbac7d5f1273282331b9b1866c0479`, fixed inputs/noise, and strict
  `rtol=1e-3, atol=1e-3` pointwise checks. The first strict failure was
  `openpi_pytorch_vs_jax/image_embeddings`; this was a `FAIL`, not a successful parity result.
  Historical Carrot commit and Docker image were not recorded. Temporary golden/checkpoint
  artifacts were under the personal DFS path `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/pi05-parity-temp`.

  The recorded absolute-error distributions (P50/P90/P99/Max) and normalized squared-L2 Dice
  distance were:

  | comparison | P50 | P90 | P99 | Max | Dice |
  |---|---:|---:|---:|---:|---:|
  | Carrot vs JAX, image embeddings | 0.0126953125 | 0.0390625 | 0.09375 | 2 | 3.89943542e-05 |
  | OpenPI PyTorch vs JAX, image embeddings | 0.01171875 | 0.0390625 | 0.09375 | 2 | 3.61103573e-05 |
  | Carrot vs OpenPI PyTorch, image embeddings | 0.0115200877 | 0.0350503653 | 0.0802722502 | 1.44098663 | 2.83228836e-05 |
  | Carrot vs JAX, one step | 0.00388103724 | 0.0137346916 | 0.0446174517 | 0.0605726242 | 6.06403428e-03 |
  | OpenPI PyTorch vs JAX, one step | 0.00333453715 | 0.00906790979 | 0.0156058836 | 0.0251698494 | 1.47571711e-03 |
  | Carrot vs OpenPI PyTorch, one step | 0.00362247229 | 0.0137272969 | 0.037278648 | 0.0443056226 | 4.44139097e-03 |

  Relative-error P50/P90/P99/Max for the one-step rows were respectively
  `0.0832973712/0.450216938/3.07924464/107.776151`,
  `0.0705956231/0.356180834/2.20201406/127.013598`, and
  `0.0772736456/0.421038152/3.9895413/92.0454943`; near-zero references inflate the
  relative maximum. The official PyTorch path itself differed from JAX beginning at image
  embeddings, while Carrot and official PyTorch were closer there but still diverged at AdaRMS,
  the first `v_t`, and the one-step output. The strict pointwise gate therefore remains open.

- A separate official PyTorch parity gate passed as `1 passed in 57.62s`, with zero numerical
  difference against the official PyTorch golden; this is a successful PyTorch-vs-Carrot result
  distinct from the failed three-way JAX diagnostic above. Historical image and exact Carrot
  commit were not recorded.

- Cleanup of the old OpenGiga/LeRobot parity tests confirmed the official PyTorch parity test
  still collected; it is maintenance evidence, not an additional numerical parity run.

## 2026-09-26 H20 rerun: FAIL at image embeddings

The old `${MY_DFS}/pi05-parity-temp` directory was absent. The official JAX
checkpoint at `${MY_DFS}/hf-hub/Physical-Intelligence/pi05_base/params` and the
converted PyTorch checkpoint at `${MY_DFS}/hf-hub/Physical-Intelligence/pi05_base_pytorch`
were present, so both goldens were regenerated in order. The remote OpenPI copy
has no `.git`; its `src/openpi` matched the local checkout by `rsync -acn`, and
`git diff 215abfb..89d9220 -- src packages pyproject.toml` was empty. Thus the
model source used for generation matches the required `215abfb` revision even
though the local checkout's later commit changed `examples/libero/main.py`.

The JAX generation used `/opt/venvs/openpi-libero`, the official `pi05_base`
checkpoint, and seed `20260914`; the PyTorch generation used the same environment,
the JAX NPZ inputs, and `pi05_base_pytorch`. Both commands exited 0. The files are:

- `${MY_DFS}/experiments/carrot/test-runs/pi05-pr32-20260926T043636Z/pi05-openpi-jax-golden.npz`
  (SHA-256 `55383fa14e4a8efe5898cdb8372d189e3ba348f6210287e4524ab0778eada3ac`)
- `${MY_DFS}/experiments/carrot/test-runs/pi05-pr32-20260926T043636Z/pi05-openpi-pytorch-golden.npz`
  (SHA-256 `9fc187d9ce130124db050e096c09a7327d585ca98ccd9a03e3647185aa963acd`)

The Carrot test used `/opt/venvs/carrot`, H20 GPU 0, merged source commit
`eddfffbda2c2980bf1563266da8275093d137bc2`, and those two NPZs. It ran
for `59.06s` and exited 1. Representative weights were BF16 bitwise equal;
language and suffix embeddings and AdaRMS condition matched exactly. The first
strict failure was `carrot_vs_openpi_pytorch/image_embeddings`: absolute
P50/P90/P99/Max `0.0078125/0.03125/0.0625/1.5`, FP64 Dice
`1.92318865e-05`, and 1,328,601 of 1,572,864 elements outside the original
`rtol=1e-3, atol=1e-3` pointwise tolerance. One-step output maximum absolute
error was `0.0144917965`. This is a real parity failure after golden generation,
not a missing-asset skip. No tolerance or model code was changed. Full output:
`${MY_DFS}/experiments/carrot/test-runs/pi05-pr32-20260926T043636Z/openpi_parity_failed.log`.

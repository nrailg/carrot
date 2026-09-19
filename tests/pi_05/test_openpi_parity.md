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

Generate deterministic JAX and PyTorch reference outputs from the same official OpenPI revision:

```bash
cd /path/to/openpi
uv run python /path/to/carrot/tests/pi_05/generate_openpi_jax_golden.py \
  --checkpoint /path/to/official/pi05_base \
  --openpi-commit "$(git rev-parse HEAD)" \
  --output /path/to/pi05-openpi-jax-golden.npz

uv run python /path/to/carrot/tests/pi_05/generate_openpi_pytorch_golden.py \
  --checkpoint /path/to/converted/pytorch/checkpoint \
  --jax-golden /path/to/pi05-openpi-jax-golden.npz \
  --openpi-commit "$(git rev-parse HEAD)" \
  --output /path/to/pi05-openpi-pytorch-golden.npz
```

Run the Carrot parity gate:

```bash
cd /path/to/carrot
export CARROT_PI05_OPENPI_GOLDEN=/path/to/pi05-openpi-jax-golden.npz
export CARROT_PI05_OPENPI_PYTORCH_GOLDEN=/path/to/pi05-openpi-pytorch-golden.npz
export CARROT_PI05_OPENPI_PYTORCH_CHECKPOINT=/path/to/converted/pytorch/checkpoint
export MY_DFS=/path/to/current/personal-dfs
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

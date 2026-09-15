# PI0.5 official OpenPI PyTorch parity

The parity test is opt-in because it loads the converted official PyTorch checkpoint and requires
a GPU. It treats official OpenPI PyTorch as the strict gate; JAX outputs are retained only for
diagnostics.

Generate deterministic JAX and PyTorch reference outputs from the same official OpenPI revision:

```bash
cd /path/to/openpi
uv run python /path/to/carrot/tests/pi05_parity/generate_openpi_jax_golden.py \
  --checkpoint /path/to/official/pi05_base \
  --openpi-commit "$(git rev-parse HEAD)" \
  --output /path/to/pi05-openpi-jax-golden.npz

uv run python /path/to/carrot/tests/pi05_parity/generate_openpi_pytorch_golden.py \
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
CUDA_VISIBLE_DEVICES=0 pytest -v -s --timeout=1800 tests/test_pi05_openpi_parity.py
```

The test checks strict checkpoint loading, representative language/action/vision weights,
intermediate embeddings, the first denoising velocity, and one Euler sampling step. It does not
test tokenization, normalization, action unnormalization, or simulator behavior.

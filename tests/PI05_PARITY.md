# PI0.5 checkpoint and inference parity

These tests are opt-in because they read two 13–14 GB checkpoints and the inference test requires a GPU. The OpenGiga checkpoint must be converted locally from the official OpenPI JAX checkpoint; a third-party converted checkpoint is not a parity oracle.

On Gemini, download the official checkpoint first:

```bash
gsutil cp -r \
  gs://openpi-assets/checkpoints/pi05_base \
  /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/Physical-Intelligence/

bash tests/pi05_parity/convert_open_giga_pi05.sh
```

The conversion script pins the expected GigaModels commit and uses the colocated Miical tokenizer only as a tokenizer artifact. It does not read Miical model weights.

```bash
export CARROT_PI05_LEROBOT_CHECKPOINT=/path/to/lerobot/pi05_base
export CARROT_PI05_OPEN_GIGA_CHECKPOINT=/path/to/self-converted/torch_pi05_base

pytest -v -s --timeout=1800 tests/test_pi05_checkpoint_parity.py
CUDA_VISIBLE_DEVICES=0 pytest -v -s --timeout=1800 tests/test_pi05_inference_parity.py
```

`test_pi05_checkpoint_parity.py` maps every parameter consumed by the OpenGiga architecture and requires bitwise equality. The only expected unused LeRobot parameter is the expert language-model head, which OpenGiga does not instantiate and PI0.5 action inference does not consume.

`test_pi05_inference_parity.py` loads the LeRobot checkpoint into both implementations, uses fixed preprocessed images, tokens, masks, state, and Gaussian noise, then compares one denoising step and the complete ten-step Euler sample. It does not test tokenization, normalization, or action unnormalization.

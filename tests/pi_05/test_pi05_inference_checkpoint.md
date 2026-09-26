# PI0.5 real-checkpoint inference

- Python: `test_pi05_inference_checkpoint.py` checks finite sampled actions and fixed-noise reproducibility.
- Run: set `MY_DFS` and `CARROT_PI05_INFERENCE_CHECKPOINT` to an existing RoboTwin SFT export,
  then `bash tests/pi_05/test_pi05_inference_checkpoint.sh`.
- Missing export or CUDA now fails the pytest case; this test has no RoboTwin export available
  in the 2026-09-26 H20 session and was not reported as passed.
- Historical inference work marked this GPU checkpoint smoke `NOT RUN`; no image tag or Carrot
  commit was recorded for a completed run.
- Status after relocation: `NOT RUN`.
- Carrot commit at relocation: `72d8168540d74e890358003cbbb4886f5c7ca0e4`.
- Docker image for next GPU run: `wepsdl/carrot:v1.10-driver-575.57.08-cuda-12.8` (planned, not historical).

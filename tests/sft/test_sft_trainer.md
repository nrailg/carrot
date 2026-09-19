# SFT trainer smoke

- Python: `test_sft_trainer.py` checks scheduler floor, failed-peer teardown, and an opt-in two-step PI0.5 Ray GPU smoke.
- Run: set `MY_DFS`, `CARROT_PI05_MODEL_PATH`, and `CARROT_ROBOTWIN_ROOT` to existing assets;
  start Ray according to `carrot-test`, then `bash tests/sft/test_sft_trainer.sh`.
- The runner checks assets before testing; the Python smoke no longer embeds a historical CephFS path.
- Historical image tag, Carrot commit, and separate result: not recorded.
- Status after relocation: `NOT RUN`.
- Carrot commit at relocation: `72d8168540d74e890358003cbbb4886f5c7ca0e4`.
- Docker image for next GPU run: `wepsdl/carrot:v1.10-driver-575.57.08-cuda-12.8` (planned, not historical).

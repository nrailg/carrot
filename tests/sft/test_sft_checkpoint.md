# SFT checkpoint format

- Python: `test_sft_checkpoint.py` checks model/optimizer/scheduler/step round trip and export artifacts.
- Run: `bash tests/sft/test_sft_checkpoint.sh` after resolving `MY_DFS`.
- Historical checkpoint work reports a combined Gemini regression `PASS`; it does not identify
  this file's image tag or exact Carrot commit.
- Status after relocation: `NOT RUN`.
- Carrot commit at relocation: `72d8168540d74e890358003cbbb4886f5c7ca0e4`.
- Docker image for next GPU run: `wepsdl/carrot:v1.10-driver-575.57.08-cuda-12.8` (planned, not historical).

## Migrated historical record

- The combined checkpoint regression was `5 passed, 3 warnings in 31.61s`; warnings were
  single-process DCP distributed-initialization notices. It also verified OpenPI root artifacts,
  optimizer/scheduler/step restore, and strict PI0.5 reload. No per-file Docker image or Carrot
  commit was recorded, so neither is inferred here.

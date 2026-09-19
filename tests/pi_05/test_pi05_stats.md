# PI0.5 statistics contract

- Python: `test_pi05_stats.py` checks future action windows relative to the current state.
- Run: `bash tests/pi_05/test_pi05_stats.sh` after resolving `MY_DFS`.
- Historical inference work mentions this statistics regression but does not provide a separate
  outcome, image tag, or Carrot commit.
- Status after relocation: `NOT RUN`.
- Carrot commit at relocation: `72d8168540d74e890358003cbbb4886f5c7ca0e4`.
- Docker image for next GPU run: `wepsdl/carrot:v1.10-driver-575.57.08-cuda-12.8` (planned, not historical).

## Migrated historical record

- This file was `BLOCKED` at collection because the local environment lacked `datasets`; no
  statistics result was claimed. The recorded command was the local pytest regression covering
  inference, checkpoint, and modeling tests. Historical Docker image and Carrot commit were not
  recorded.

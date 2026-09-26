# Simulator sidecar isolation

- Python: `test_simulator_sidecar.py` describes Ray supervision of simulators in distinct venvs.
- Run: `bash tests/simulator_sidecar/test_simulator_sidecar.sh` after resolving `MY_DFS`.
- Current module status: `SKIP` at module level; do not report that as simulator validation.
- If this explicit module-level skip is removed later, missing MuJoCo interpreter paths fail
  the real-version test rather than silently skipping it.
- Historical image tag, Carrot commit, and completed result: not recorded.
- Carrot commit at relocation: `72d8168540d74e890358003cbbb4886f5c7ca0e4`.
- Docker image for next GPU run: `wepsdl/carrot:v1.10-driver-575.57.08-cuda-12.8` (planned, not historical).

## 2026-09-26 tracked-file run

Both collected cases retained the explicit module-level `暂时跳过` marker;
the result was `2 skipped`. This is the only remaining skip in the tracked-file
pytest run after asset prerequisite skips became failures. Output:
`${MY_DFS}/experiments/carrot/test-runs/pi05-pr32-20260926T043636Z/tracked_pytest_failfast.log`.

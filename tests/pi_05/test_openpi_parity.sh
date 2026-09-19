#!/usr/bin/env bash
set -Eeuo pipefail

CARROT_DIR="${MY_DFS:?set MY_DFS to the current personal DFS root}/work/carrot"
PYTHON_BIN=/opt/venvs/carrot/bin/python

[[ -f "${CARROT_DIR}/tests/pi_05/test_openpi_parity.py" ]]
[[ -x "$PYTHON_BIN" ]]
[[ -s "${CARROT_PI05_OPENPI_GOLDEN:?set the JAX golden path}" ]]
[[ -s "${CARROT_PI05_OPENPI_PYTORCH_GOLDEN:?set the PyTorch golden path}" ]]
[[ -f "${CARROT_PI05_OPENPI_PYTORCH_CHECKPOINT:?set the checkpoint path}/config.json" ]]

source /opt/venvs/carrot/bin/activate
cd "$CARROT_DIR"
export PYTHONPATH="$PWD/src:$PWD/tests:${PYTHONPATH:-}"
test "$(command -v python)" = "$PYTHON_BIN"
python -m pytest -v -s --timeout=1800 tests/pi_05/test_openpi_parity.py

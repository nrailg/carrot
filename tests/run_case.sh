#!/usr/bin/env bash
set -Eeuo pipefail

CASE_PATH="${1:?pass a repository-relative test file}"
shift
CARROT_DIR="${MY_DFS:?set MY_DFS to the current personal DFS root}/work/carrot"
[[ -f "${CARROT_DIR}/${CASE_PATH}" ]]
[[ -x /opt/venvs/carrot/bin/python ]]

source /opt/venvs/carrot/bin/activate
cd "$CARROT_DIR"
export PYTHONPATH="$PWD/src:$PWD/tests:${PYTHONPATH:-}"
test "$(command -v python)" = /opt/venvs/carrot/bin/python
python -m pytest -v -s --timeout=1800 "$CASE_PATH" "$@"

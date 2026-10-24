#!/usr/bin/env bash
set -Eeuo pipefail
: "${MY_DFS:?confirm H20 DFS}"
: "${COLLECTION_DIR:?atomic verified copy from PRO}"
: "${CARROT_SOURCE_DIR:?frozen experiment source}"
: "${RAY_ADDRESS:?verified eight-GPU cluster}"
: "${RUN_DIR:?new formal training directory}"
source /opt/venvs/carrot/bin/activate
echo 'Waiting for verified collection; optimizer steps have not started.'
while [[ ! -f "$COLLECTION_DIR/manifest.json" ]]; do
  [[ ! -f "$COLLECTION_DIR.failed" ]] || { echo 'Collection/copy failed'; exit 1; }
  sleep 30
done
bash /root/dguard/dguard.sh stop 120 --local
trap 'bash /root/dguard/dguard.sh on --local' EXIT
sleep 5
# Do not consume a GPU acquired by another job while this run waited for data.
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || {
  echo 'GPU has active compute processes; stop before training.'
  exit 1
}
export DRAW_ROOT="$COLLECTION_DIR/demos"
export TEACHER_GATE="$COLLECTION_DIR/teacher-gate.json"
bash "$CARROT_SOURCE_DIR/recipes/pi05_libero_draw_sft_transfer/run.sh"

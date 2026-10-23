#!/usr/bin/env bash
set -Eeuo pipefail
ROLE="${1:?head or worker}"
[[ "$ROLE" == head || "$ROLE" == worker ]]
: "${MY_DFS:?confirm DFS separately on this resource pool}"
: "${RAY_HEAD_IP:?set the reachable H20 head node address}"
: "${CARROT_VISIBLE_GPUS:?explicit idle GPU indices for this node}"
: "${DGUARD_STOP_MINUTES:?10 for short checks, 120 for large tests, 2880 for the pilot}"
source /opt/venvs/carrot/bin/activate
cd "${MY_DFS}/work/carrot"
export PYTHONPATH="$PWD/src:$PWD/examples/isaac_workbench:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$CARROT_VISIBLE_GPUS"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
[[ -z "$(pgrep -x raylet || true)" ]] || { echo 'Existing Ray node: inspect before changing it'; exit 1; }
bash /root/dguard/dguard.sh status
bash /root/dguard/dguard.sh stop "$DGUARD_STOP_MINUTES"
ray stop --force
PORT="${RAY_PORT:-6379}"
if [[ "$ROLE" == head ]]; then
  [[ "${__HOST_IP__:?missing current node address}" == "$RAY_HEAD_IP" ]]
  ray start --head --node-ip-address="$RAY_HEAD_IP" --port="$PORT" --disable-usage-stats \
    --num-cpus=16 --object-store-memory=2147483648
else
  ray start --address="${RAY_HEAD_IP}:${PORT}" --node-ip-address="${__HOST_IP__:?}" --disable-usage-stats \
    --num-cpus=16 --object-store-memory=2147483648
fi

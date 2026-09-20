#!/usr/bin/env bash

set -Eeuo pipefail

[[ $# -eq 1 && "$1" =~ ^[0-9]+$ ]] || {
    echo "usage: $0 EXISTING_BENCHMARK_PID" >&2
    exit 2
}
[[ "${__HOST_IP__:-}" == 29.209.163.23 ]] || {
    echo "this runner must execute on mpi-1760058051-launcher" >&2
    exit 1
}

existing_pid="$1"
my_dfs="${MY_DFS:?set MY_DFS to the current personal DFS root}"
carrot_dir="$my_dfs/work/carrot"
checkpoint_root="$my_dfs/experiments/carrot/pi05_libero_sft_2k_20260919/checkpoints"
output_root="$my_dfs/benchmarks/carrot-pi05-libero-sft-four-suites-20260919"
jobs=(
    00000100:libero_10
    00001000:libero_spatial
    00001000:libero_object
)

for job in "${jobs[@]}"; do
    step="${job%%:*}"
    checkpoint="$checkpoint_root/step-$step"
    for file in model.safetensors config.json trainer_state.json \
                assets/physical-intelligence/libero/norm_stats.json; do
        [[ -s "$checkpoint/$file" ]] || {
            echo "missing checkpoint asset: $checkpoint/$file" >&2
            exit 1
        }
    done
done
[[ -f "$carrot_dir/tests/pi_05/test_libero.sh" ]] || {
    echo "missing LIBERO runner" >&2
    exit 1
}

# The existing runner uses GPU 0/1 and port 8000. Never overlap its renderer.
while [[ -r "/proc/$existing_pid/cmdline" ]] &&
      tr '\0' ' ' < "/proc/$existing_pid/cmdline" |
          grep -Fq 'run_libero_sft_benchmark.sh new 29.209.163.23'; do
    sleep 30
done

source /opt/venvs/carrot/bin/activate
cd "$carrot_dir"
export PYTHONPATH="$PWD/src:$PWD/tests:${PYTHONPATH:-}"
[[ "$(command -v python)" == /opt/venvs/carrot/bin/python ]] || {
    echo "wrong Carrot interpreter" >&2
    exit 1
}

for job in "${jobs[@]}"; do
    step="${job%%:*}"
    suite="${job#*:}"
    checkpoint="$checkpoint_root/step-$step"
    suite_output="$output_root/step-$step/$suite"
    echo "RELOCATED_START step=$step suite=$suite time=$(date -Is)"
    if MY_DFS="$my_dfs" TASK_SUITE="$suite" TASK_ID=all EPISODES=50 MIN_SUCCESSES=0 \
       DGUARD_STOP_MINUTES=360 CHECKPOINT_DIR="$checkpoint" OUTPUT_ROOT="$suite_output" \
       bash tests/pi_05/test_libero.sh; then
        echo "RELOCATED_DONE step=$step suite=$suite time=$(date -Is)"
    else
        rc=$?
        echo "RELOCATED_FAIL step=$step suite=$suite rc=$rc time=$(date -Is)" >&2
        exit "$rc"
    fi
done

echo "RELOCATED_LANE_DONE time=$(date -Is)"

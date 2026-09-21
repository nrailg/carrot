#!/usr/bin/env bash

set -Eeuo pipefail

[[ $# -eq 2 ]] || {
    echo "usage: $0 old|new EXPECTED_HOST_IP" >&2
    exit 2
}

lane="$1"
expected_host_ip="$2"
[[ "${__HOST_IP__:-}" == "$expected_host_ip" ]] || {
    echo "wrong launcher: expected $expected_host_ip, got ${__HOST_IP__:-<unset>}" >&2
    exit 1
}

case "$lane" in
    old)
        jobs=(
            00000100:libero_spatial
            00000100:libero_object
            00000100:libero_goal
            00000100:libero_10
            00001000:libero_spatial
            00001000:libero_object
        )
        ;;
    new)
        jobs=(
            00000300:libero_spatial
            00000300:libero_object
            00000300:libero_goal
            00000300:libero_10
            00001000:libero_goal
            00001000:libero_10
        )
        ;;
    *)
        echo "unknown lane: $lane" >&2
        exit 2
        ;;
esac

my_dfs="${MY_DFS:?set MY_DFS to the current personal DFS root}"
carrot_dir="$my_dfs/work/carrot"
checkpoint_root="$my_dfs/experiments/carrot/pi05_libero_sft_2k_20260919/checkpoints"
output_root="$my_dfs/benchmarks/carrot-pi05-libero-sft-four-suites-20260919"

[[ -f "$carrot_dir/tests/pi_05/test_libero.sh" ]] || {
    echo "missing LIBERO runner: $carrot_dir/tests/pi_05/test_libero.sh" >&2
    exit 1
}
[[ -f "$my_dfs/work/openpi/examples/libero/main.py" ]] || {
    echo "missing OpenPI evaluator" >&2
    exit 1
}
[[ -f "$my_dfs/hf-hub/google/paligemma-3b-pt-224/tokenizer.json" ]] || {
    echo "missing local tokenizer" >&2
    exit 1
}

# 先检查所有任务，避免跑完几套后才发现下一份 checkpoint 或输出路径有问题。
for job in "${jobs[@]}"; do
    step="${job%%:*}"
    suite="${job#*:}"
    checkpoint="$checkpoint_root/step-$step"
    suite_output="$output_root/step-$step/$suite"
    for file in model.safetensors config.json trainer_state.json \
                assets/physical-intelligence/libero/norm_stats.json; do
        [[ -s "$checkpoint/$file" ]] || {
            echo "missing checkpoint asset: $checkpoint/$file" >&2
            exit 1
        }
    done
    [[ ! -e "$suite_output" ]] || {
        echo "benchmark output already exists: $suite_output" >&2
        exit 1
    }
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
    echo "BENCHMARK_START lane=$lane step=$step suite=$suite time=$(date -Is)"
    if MY_DFS="$my_dfs" TASK_SUITE="$suite" TASK_ID=all EPISODES=50 MIN_SUCCESSES=0 \
       DGUARD_STOP_MINUTES=360 CHECKPOINT_DIR="$checkpoint" OUTPUT_ROOT="$suite_output" \
       bash tests/pi_05/test_libero.sh; then
        echo "BENCHMARK_DONE lane=$lane step=$step suite=$suite time=$(date -Is)"
    else
        rc=$?
        echo "BENCHMARK_FAIL lane=$lane step=$step suite=$suite rc=$rc time=$(date -Is)" >&2
        exit "$rc"
    fi
done

echo "BENCHMARK_LANE_DONE lane=$lane time=$(date -Is)"

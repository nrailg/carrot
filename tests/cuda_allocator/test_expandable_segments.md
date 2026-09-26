# CUDA allocator expandable segments benchmark

## Purpose

Measure the latency and memory impact of PyTorch CUDA allocator
`expandable_segments` without treating a hardware-dependent speed ratio as a correctness
assertion. The fixed-size workload is the overhead control. The dynamic workload allocates
`8 x 16 MiB`, releases them, then allocates `4 x 32 MiB` to reproduce the cross-segment
fragmentation pattern described by PyTorch.

## Metrics and assertions

- Run `expandable_segments:False` and `True` in separate processes so allocator state is not
  shared.
- Report cold wall/CUDA-event latency and steady wall/CUDA-event P50, P90, P99, mean, and max.
- Report allocated, reserved, peak, inactive-split, segment, device-allocation, retry, and OOM
  counters.
- Assert that `memory_snapshot()` observes the requested segment type, both runs use matching
  software/hardware parameters, samples are complete, and reserved bytes cover allocated bytes.
- Ratios in `summary.json` are `True / False`; values below one mean lower time or memory usage.
  No speed or memory-reduction threshold is asserted.

## Run

After resolving `MY_DFS`, syncing this checkout, checking dguard, and activating the existing
Carrot environment:

```bash
MY_DFS="$MY_DFS" bash tests/cuda_allocator/test_expandable_segments.sh
```

Optional controls are `CUDA_DEVICE`, `WARMUP`, `ITERATIONS`, and `RUN_DIR`. Defaults are logical
CUDA device 0, 5 warmups, 30 measured iterations, and a timestamped directory below
`${MY_DFS}/benchmarks/carrot-expandable-segments/`.

## References

- PyTorch CUDA memory notes:
  <https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage>
- PyTorch allocator implementation and `Note [Expandable Segments]`:
  <https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp>
- PyTorch allocator fragmentation analysis:
  <https://docs.pytorch.org/devlogs/eager/2026-06-01-cuda-caching-allocator/>

## Result

- Status: `PASS` (2026-09-22 rerun)
- Docker image: not recorded
- Carrot commit: `aa7e1717cbda036f90eb029a66edcf2eeb0518e5`; the remote checkout had no
  `.git`, so the executed Python and shell files were matched to this commit by SHA256.
- Evidence directory:
  `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/benchmarks/carrot-expandable-segments/20260922-040631`

### Initial preflight

- Environment preflight: `/opt/venvs/carrot/bin/python`, PyTorch `2.11.0+cu128`, CUDA
  `12.8`, 8 x NVIDIA H20.
- The first attempt stopped before execution after detecting the expected dguard health-check
  load. The rerun used `bash /root/dguard/dguard.sh stop 10 --local`; dguard reported a scheduled
  restore and all GPUs reached 0% utilization before measurement.

### Successful rerun

- Command: `MY_DFS="$MY_DFS" bash tests/cuda_allocator/test_expandable_segments.sh`
- Exit code: `0`; warmup: 5; measured iterations: 30; logical device: `cuda:0`.
- Both allocator modes matched `memory_snapshot()`. Logs contained no warning, traceback, CUDA
  error, allocation retry, or OOM.

| Workload | Steady wall P50, False | Steady wall P50, True | P50 change | Reserved, False | Reserved, True | Reserved change | Segments |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed `8 x 16 MiB` | 0.092071 ms | 0.092700 ms | +0.68% | 128 MiB | 140 MiB | +9.38% | 8 -> 1 |
| Dynamic `8 x 16 -> 4 x 32 MiB` | 0.138593 ms | 0.140368 ms | +1.28% | 256 MiB | 140 MiB | -45.31% | 12 -> 1 |

CUDA-event P50 increased by 1.37% for the fixed workload and 1.39% for the dynamic workload.
The single cold sample was 15.37% slower for fixed sizes and 49.86% slower for dynamic sizes;
these cold ratios are directional only because each mode has one sample. On this synthetic H20
run, expandable segments traded about 1% steady latency for substantially lower reserved memory
only when allocation sizes changed.

## 2026-09-26 H20 rerun: PASS

On `mpi-launcher@mpi-1759754893-launcher` with the synced Carrot commit
`eddfffbda2c2980bf1563266da8275093d137bc2`, dguard paused and H20 GPU 0
idle, run:

```bash
export MY_DFS=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu
export RUN_DIR="$MY_DFS/experiments/carrot/test-runs/pi05-pr32-20260926T043636Z/cuda_allocator"
bash tests/cuda_allocator/test_expandable_segments.sh
```

The script exited 0,
using 5 warmups and 30 measured iterations in each allocator mode. The dynamic
workload reserved 256 MiB with expandable segments off and 140 MiB with them on
(ratio `0.546875`); steady wall P50 ratio was `0.99795`. This benchmark does not
assert a performance threshold. JSON, per-mode logs, and `summary.log` are in
the `RUN_DIR` above; full wrapper output is
`${MY_DFS}/experiments/carrot/test-runs/pi05-pr32-20260926T043636Z/cuda_allocator_run.log`.

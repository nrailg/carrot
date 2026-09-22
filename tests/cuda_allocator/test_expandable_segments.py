from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

MIB = 1024 * 1024


def _allocate(count: int, size_mib: int, device: torch.device) -> list[torch.Tensor]:
    tensors = [
        torch.empty(size_mib * MIB, dtype=torch.uint8, device=device) for _ in range(count)
    ]
    for tensor in tensors:
        tensor.fill_(1)
    return tensors


def _fixed_workload(device: torch.device) -> None:
    tensors = _allocate(count=8, size_mib=16, device=device)
    tensors.clear()


def _dynamic_workload(device: torch.device) -> None:
    small_tensors = _allocate(count=8, size_mib=16, device=device)
    small_tensors.clear()
    large_tensors = _allocate(count=4, size_mib=32, device=device)
    large_tensors.clear()


def _measure_once(
    workload: Callable[[torch.device], None], device: torch.device
) -> tuple[float, float]:
    torch.cuda.synchronize(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start = time.perf_counter_ns()
    start_event.record()
    workload(device)
    end_event.record()
    torch.cuda.synchronize(device)
    wall_ms = (time.perf_counter_ns() - start) / 1_000_000
    return wall_ms, start_event.elapsed_time(end_event)


def _percentiles(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)

    def nearest_rank(percentile: float) -> float:
        index = round((len(ordered) - 1) * percentile)
        return ordered[index]

    return {
        "mean": statistics.fmean(ordered),
        "p50": nearest_rank(0.50),
        "p90": nearest_rank(0.90),
        "p99": nearest_rank(0.99),
        "max": ordered[-1],
    }


def _memory_metrics(device: torch.device) -> dict[str, int]:
    stats = torch.cuda.memory_stats(device)
    snapshot = [
        segment
        for segment in torch.cuda.memory_snapshot()
        if segment["device"] == device.index
    ]
    return {
        "allocated_bytes": stats["allocated_bytes.all.current"],
        "reserved_bytes": stats["reserved_bytes.all.current"],
        "peak_allocated_bytes": stats["allocated_bytes.all.peak"],
        "peak_reserved_bytes": stats["reserved_bytes.all.peak"],
        "inactive_split_bytes": stats["inactive_split_bytes.all.current"],
        "segment_count": len(snapshot),
        "expandable_segment_count": sum(bool(segment["is_expandable"]) for segment in snapshot),
        "snapshot_segment_bytes": sum(int(segment["total_size"]) for segment in snapshot),
        "num_device_alloc": stats["num_device_alloc"],
        "num_device_free": stats["num_device_free"],
        "num_alloc_retries": stats["num_alloc_retries"],
        "num_ooms": stats["num_ooms"],
    }


def _run_workload(
    name: str,
    workload: Callable[[torch.device], None],
    device: torch.device,
    expected_expandable: bool,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    # 清空缓存后测量首次建段成本，避免前一个 workload 的 segment 污染 cold 数据。
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.reset_accumulated_memory_stats(device)
    cold_wall_ms, cold_cuda_ms = _measure_once(workload, device)
    cold_memory = _memory_metrics(device)

    # 预热建立可复用缓存，steady 数据反映已有 segment 下的重复分配成本。
    for _ in range(warmup):
        workload(device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.reset_accumulated_memory_stats(device)

    # 同时采集 host wall time 和 CUDA event，区分 allocator host 开销与 GPU 工作。
    wall_samples = []
    cuda_samples = []
    for _ in range(iterations):
        wall_ms, cuda_ms = _measure_once(workload, device)
        wall_samples.append(wall_ms)
        cuda_samples.append(cuda_ms)
    memory = _memory_metrics(device)

    # memory_snapshot 必须证明本进程实际采用了请求的 allocator 配置。
    assert memory["segment_count"] > 0, f"{name}: allocator produced no segments"
    if expected_expandable:
        assert memory["expandable_segment_count"] > 0, (
            f"{name}: expandable_segments=True produced no expandable segment"
        )
    else:
        assert memory["expandable_segment_count"] == 0, (
            f"{name}: expandable_segments=False produced an expandable segment"
        )
    assert memory["reserved_bytes"] >= memory["allocated_bytes"]
    assert len(wall_samples) == iterations
    assert len(cuda_samples) == iterations

    return {
        "name": name,
        "cold": {
            "wall_ms": cold_wall_ms,
            "cuda_ms": cold_cuda_ms,
            "memory": cold_memory,
        },
        "steady": {
            "wall_ms": _percentiles(wall_samples),
            "cuda_ms": _percentiles(cuda_samples),
            "memory": memory,
        },
    }


def _allocator_config() -> str:
    return os.environ.get(
        "PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    )


def _run_benchmark(args: argparse.Namespace) -> None:
    # 先校验运行环境，避免 CPU fallback 或错误配置生成看似有效的性能报告。
    assert torch.cuda.is_available(), "CUDA is required"
    expected_expandable = args.expected_expandable == "true"
    allocator_config = _allocator_config()
    expected_token = f"expandable_segments:{str(expected_expandable)}"
    assert expected_token.lower() in allocator_config.lower(), (
        f"allocator config {allocator_config!r} does not contain {expected_token!r}"
    )
    assert args.warmup >= 1
    assert args.iterations >= 2

    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    torch.cuda.init()

    # 固定尺寸是额外开销控制组，动态尺寸复现跨 segment 无法合并的碎片模式。
    workloads = [
        _run_workload(
            "fixed_8x16_mib",
            _fixed_workload,
            device,
            expected_expandable,
            args.warmup,
            args.iterations,
        ),
        _run_workload(
            "dynamic_8x16_then_4x32_mib",
            _dynamic_workload,
            device,
            expected_expandable,
            args.warmup,
            args.iterations,
        ),
    ]

    # 报告记录软硬件和参数，使 True/False 结果具备可比性与复现条件。
    result = {
        "schema_version": 1,
        "expected_expandable": expected_expandable,
        "allocator_config": allocator_config,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "device": str(device),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "workloads": workloads,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


def _workloads_by_name(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {workload["name"]: workload for workload in result["workloads"]}


def _ratio(numerator: float, denominator: float) -> float:
    assert denominator > 0
    return numerator / denominator


def _compare(args: argparse.Namespace) -> None:
    # 读取独立进程结果，确保 baseline 与 expandable 没有共享 allocator 状态。
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    expandable = json.loads(Path(args.expandable).read_text(encoding="utf-8"))
    assert baseline["expected_expandable"] is False
    assert expandable["expected_expandable"] is True
    comparable_keys = (
        "schema_version",
        "torch_version",
        "cuda_version",
        "gpu_name",
        "warmup",
        "iterations",
    )
    for key in comparable_keys:
        assert baseline[key] == expandable[key], (
            f"incomparable field {key}: {baseline[key]!r} != {expandable[key]!r}"
        )

    baseline_workloads = _workloads_by_name(baseline)
    expandable_workloads = _workloads_by_name(expandable)
    assert baseline_workloads.keys() == expandable_workloads.keys()

    # 比值统一为 expandable/baseline；小于 1 表示时间或显存占用下降。
    comparisons = []
    for name, baseline_workload in baseline_workloads.items():
        expandable_workload = expandable_workloads[name]
        comparisons.append(
            {
                "name": name,
                "cold_wall_ratio": _ratio(
                    expandable_workload["cold"]["wall_ms"],
                    baseline_workload["cold"]["wall_ms"],
                ),
                "steady_wall_p50_ratio": _ratio(
                    expandable_workload["steady"]["wall_ms"]["p50"],
                    baseline_workload["steady"]["wall_ms"]["p50"],
                ),
                "steady_wall_p99_ratio": _ratio(
                    expandable_workload["steady"]["wall_ms"]["p99"],
                    baseline_workload["steady"]["wall_ms"]["p99"],
                ),
                "steady_cuda_p50_ratio": _ratio(
                    expandable_workload["steady"]["cuda_ms"]["p50"],
                    baseline_workload["steady"]["cuda_ms"]["p50"],
                ),
                "reserved_bytes_ratio": _ratio(
                    expandable_workload["steady"]["memory"]["reserved_bytes"],
                    baseline_workload["steady"]["memory"]["reserved_bytes"],
                ),
                "baseline": baseline_workload,
                "expandable": expandable_workload,
            }
        )

    summary = {
        "schema_version": 1,
        "ratio_definition": "expandable_segments_true / expandable_segments_false",
        "torch_version": baseline["torch_version"],
        "cuda_version": baseline["cuda_version"],
        "gpu_name": baseline["gpu_name"],
        "warmup": baseline["warmup"],
        "iterations": baseline["iterations"],
        "comparisons": comparisons,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--expected-expandable", choices=("false", "true"), required=True)
    benchmark.add_argument("--device", type=int, default=0)
    benchmark.add_argument("--warmup", type=int, default=5)
    benchmark.add_argument("--iterations", type=int, default=30)
    benchmark.add_argument("--output", required=True)
    benchmark.set_defaults(handler=_run_benchmark)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--expandable", required=True)
    compare.add_argument("--output", required=True)
    compare.set_defaults(handler=_compare)
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = _parse_args()
    parsed_args.handler(parsed_args)

"""Profile a small Ray data worker and a PyTorch learner in Nsight Systems."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import ray
import torch


@contextmanager
def nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


@ray.remote(num_cpus=1)
class DataWorker:
    def pid(self) -> int:
        return os.getpid()

    def batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        with nvtx_range("data: prepare batch"):
            generator = torch.Generator().manual_seed(7)
            features = torch.randn(1024, 64, generator=generator)
            targets = 0.5 * features[:, :1] - features[:, 1:2]
            time.sleep(0.05)
            return features, targets


@ray.remote(num_cpus=1)
class Learner:
    def __init__(self, cpu_only: bool) -> None:
        self.device = torch.device("cpu" if cpu_only else "cuda")
        torch.manual_seed(7)
        self.model = torch.nn.Linear(64, 1).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05)

    def pid(self) -> int:
        return os.getpid()

    def train_step(self, features: torch.Tensor, targets: torch.Tensor) -> float:
        with nvtx_range("learner: train step"):
            features = features.to(self.device)
            targets = targets.to(self.device)
            for _ in range(20):
                with nvtx_range("learner: forward backward optimizer"):
                    loss = torch.nn.functional.mse_loss(self.model(features), targets)
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            return float(loss.item())


def wait_for_reports(output_dir: Path, pids: dict[str, int]) -> dict[str, str]:
    reports = {role: output_dir / f"{role}_{pid}.nsys-rep" for role, pid in pids.items()}
    pending = dict(reports)
    deadline = time.monotonic() + 60
    while pending and time.monotonic() < deadline:
        for role, report in list(pending.items()):
            if report.is_file() and report.with_suffix(".sqlite").is_file():
                print(f"{role}: {report}")
                del pending[role]
        if pending:
            time.sleep(1)
    if pending:
        raise RuntimeError(f"Nsight reports missing for {pending}")
    return {role: str(report) for role, report in reports.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/ray_nsys_output"))
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--ray-temp-dir", type=Path)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--no-profile", action="store_true")
    args = parser.parse_args()

    if shutil.which("nsys") is None and not args.no_profile:
        parser.error("nsys is not on PATH")
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not args.cpu_only and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --cpu-only for an NVTX-only demo")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ray_temp_dir = (args.ray_temp_dir or Path(f"/tmp/ray_nsys_train_{os.getpid()}")).resolve()
    nsight = {
        "trace": "cuda,nvtx",
        "sample": "none",
        "cpuctxsw": "none",
        "cuda-flush-interval": "1000",
    }

    def runtime_env(role: str) -> dict:
        if args.no_profile:
            return {}
        return {"nsight": {**nsight, "o": str(output_dir / f"{role}_%p"), "export": "sqlite"}}

    ray.init(address="local", _temp_dir=str(ray_temp_dir), include_dashboard=False)
    try:
        data = DataWorker.options(runtime_env=runtime_env("data")).remote()
        learner = Learner.options(
            num_gpus=0 if args.cpu_only else 1,
            runtime_env=runtime_env("learner"),
        ).remote(args.cpu_only)
        pids = {"data": ray.get(data.pid.remote()), "learner": ray.get(learner.pid.remote())}
        losses = []
        for step in range(args.steps):
            features, targets = ray.get(data.batch.remote())
            loss = ray.get(learner.train_step.remote(features, targets))
            losses.append(loss)
            print(f"step={step + 1} loss={loss:.6f}", flush=True)
        # Let actor processes exit cleanly so Nsight can flush CUDA activity.
        exits = [data.__ray_terminate__.remote(), learner.__ray_terminate__.remote()]
        _, pending = ray.wait(exits, num_returns=len(exits), timeout=30)
        if pending:
            raise RuntimeError("Ray actors did not exit within 30 seconds")
        reports = wait_for_reports(output_dir, pids) if not args.no_profile else {}
    finally:
        ray.shutdown()
    if reports:
        print("Open the data and learner reports together in Nsight Systems Multi-report view.")
    (output_dir / "run.json").write_text(
        json.dumps(
            {"pids": pids, "losses": losses, "profiled": not args.no_profile, "reports": reports},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

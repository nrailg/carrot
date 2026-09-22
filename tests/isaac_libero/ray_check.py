"""Run the real simulator acceptance inside a Carrot Ray worker on Python 3.12."""

import argparse
import json
import os
import platform
import runpy
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any, override

from carrot.distributed import Cluster, PlacementSpec, RolePlacement, Worker


class IsaacAcceptanceWorker(Worker):
    @override
    def setup(self) -> None:
        self.context: dict[str, Any] | None = None

    def run(self, output: str) -> dict[str, Any]:
        # 在 Ray actor 本进程启动 Kit，验证相同解释器下的真实仿真和 PPO。
        script = Path(__file__).with_name("run_isaac_libero.py")
        previous_argv = sys.argv
        sys.argv = [
            str(script),
            "--headless",
            "--enable_cameras",
            "--num_envs",
            "4",
            "--steps",
            "32",
            "--output",
            output,
            "--no_fast_shutdown",
            "--kit_args=--/renderer/multiGpu/enabled=false",
        ]
        try:
            self.context = runpy.run_path(str(script), run_name="isaac_acceptance")
            self.context["main"]()
        finally:
            sys.argv = previous_argv
        result = json.loads((Path(output) / "result.json").read_text())
        return {"result": result, "worker": self.identity()}

    def identity(self) -> dict[str, Any]:
        return {
            "python": platform.python_version(),
            "executable": sys.executable,
            "ray": version("ray"),
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        }

    @override
    def teardown(self) -> None:
        if self.context is not None:
            self.context["simulation_app"].close()


def main() -> None:
    # 仅连接已启动的单 GPU 测试集群，避免意外占用其他训练资源。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert sys.version_info[:2] == (3, 12)
    assert os.environ["RAY_ADDRESS"]
    assert not (args.output / "result.json").exists()
    with Cluster(namespace="isaac-libero-py312") as cluster:
        cluster.reserve("sim", PlacementSpec(cpus_per_bundle=2, gpus_per_bundle=1))
        workers = cluster.launch(
            "sim",
            IsaacAcceptanceWorker,
            placement=RolePlacement(pool="sim", cpus_per_actor=2, gpus_per_actor=1),
        )
        result = workers.call("run", str(args.output)).wait(timeout=600)[0]
        # 再次调用同一 actor，防止把进程提前退出误判为 Ray 集成成功。
        identity = workers.call("identity").wait(timeout=30)[0]
        assert identity == result["worker"]
        assert identity["python"] == platform.python_version()
        assert identity["ray"] == version("ray")
        assert identity["executable"] == sys.executable
        assert result["result"]["status"] == "PASS"
    (args.output / "ray_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PASS: Python 3.12 Isaac rollout and PPO in a live Carrot Ray worker", flush=True)


if __name__ == "__main__":
    main()

import argparse
import json
import os
import platform
import runpy
import sys
from importlib.metadata import version
from pathlib import Path
from typing import override

from carrot.distributed import Cluster, PlacementSpec, RolePlacement, Worker


class SimulatorWorker(Worker):
    @override
    def setup(self) -> None:
        self.context = None

    def run(self, assets: str, output: str) -> dict:
        previous = sys.argv
        sys.argv = [
            "run_arena_libero.py",
            "--asset-root",
            assets,
            "--output",
            output,
            "--headless",
            "--enable_cameras",
            "--kit_args=--/renderer/multiGpu/enabled=false",
        ]
        try:
            self.context = runpy.run_path(
                str(Path(__file__).with_name("run_arena_libero.py")), run_name="arena_acceptance"
            )
            self.context["main"]()
            return json.loads((Path(output) / "result.json").read_text())
        finally:
            sys.argv = previous

    def identity(self) -> dict:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        physical_gpu = (
            visible.split(",")[self.local_rank].strip() if visible else str(self.local_rank)
        )
        return {
            "python": platform.python_version(),
            "executable": sys.executable,
            "ray": version("ray"),
            "pid": os.getpid(),
            "cuda_visible_devices": visible,
            "local_rank": self.local_rank,
            "physical_gpu": physical_gpu,
        }

    @override
    def teardown(self) -> None:
        # Kit owns this process; Cluster terminates it after the environment has closed.
        self.context = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert sys.version_info[:2] == (3, 12)
    assert os.environ.get("RAY_ADDRESS"), "Start the dedicated Ray head before running this check"
    env_vars = {
        key: os.environ[key]
        for key in (
            "PYTHONPATH",
            "PATH",
            "VIRTUAL_ENV",
            "LD_LIBRARY_PATH",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "no_proxy",
        )
        if key in os.environ
    }
    env_vars["OMNI_KIT_ACCEPT_EULA"] = "YES"
    with Cluster(namespace="arena-libero-validation", env_vars=env_vars) as cluster:
        cluster.reserve("sim", PlacementSpec(cpus_per_bundle=2, gpus_per_bundle=1))
        group = cluster.launch(
            "simulator",
            SimulatorWorker,
            placement=RolePlacement(pool="sim", cpus_per_actor=2, gpus_per_actor=1),
        )
        before = group.call("identity").wait()[0]
        assert before["physical_gpu"] == "0", f"This H20 acceptance run requires GPU 0: {before}"
        result = group.call("run", args.asset_root, str(args.output)).wait(timeout=900)[0]
        after = group.call("identity").wait()[0]
        assert before == after
        assert after["python"] == platform.python_version()
        assert after["executable"] == sys.executable
        assert after["ray"] == version("ray")
        assert result["status"] == "PASS"
    (args.output / "ray_result.json").write_text(
        json.dumps({"status": "PASS", "worker": after}, indent=2)
    )


if __name__ == "__main__":
    main()

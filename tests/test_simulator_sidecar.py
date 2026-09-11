"""Minimal proof that Ray supervisors can drive simulators in separate venvs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import venv
from pathlib import Path
from typing import Any

from carrot.distributed import Cluster, PlacementSpec, RolePlacement, Worker

_SIDECAR_PROGRAM = r"""
import json
import os
import sys

backend = os.environ["CARROT_SIM_BACKEND"]
state = 0

for line in sys.stdin:
    request = json.loads(line)
    operation = request["op"]
    if operation == "health":
        response = {
            "backend": backend,
            "executable": sys.executable,
            "prefix": sys.prefix,
            "protocol_version": 1,
        }
    elif operation == "reset":
        state = request["seed"]
        response = {"observation": {"state": state}}
    elif operation == "step":
        state += request["action"]
        response = {
            "observation": {"state": state},
            "reward": float(state),
            "terminated": False,
            "truncated": False,
        }
    elif operation == "close":
        print(json.dumps({"closed": True}), flush=True)
        break
    else:
        response = {"error": f"unknown operation: {operation}"}
    print(json.dumps(response), flush=True)
"""


class SimulatorSupervisor(Worker):
    """Ray worker in the core venv supervising one simulator sidecar."""

    def __init__(self, python_executable: str, backend: str) -> None:
        self.python_executable = python_executable
        self.backend = backend
        self.process: subprocess.Popen[str] | None = None

    def setup(self) -> None:
        environment = os.environ.copy()
        environment["CARROT_SIM_BACKEND"] = self.backend
        self.process = subprocess.Popen(
            [self.python_executable, "-u", "-c", _SIDECAR_PROGRAM],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=environment,
        )
        self.request({"op": "health"})

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("simulator sidecar is not running")
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        response = self.process.stdout.readline()
        if not response:
            stderr = ""
            if self.process.stderr is not None:
                stderr = self.process.stderr.read()
            raise RuntimeError(f"simulator sidecar exited unexpectedly: {stderr}")
        result = json.loads(response)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def health(self) -> dict[str, Any]:
        return self.request({"op": "health"})

    def reset(self, seed: int) -> dict[str, Any]:
        return self.request({"op": "reset", "seed": seed})

    def step(self, action: int) -> dict[str, Any]:
        return self.request({"op": "step", "action": action})

    def supervisor_executable(self) -> str:
        return sys.executable

    def teardown(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.request({"op": "close"})
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        self.process = None


def _create_venv(path: Path) -> str:
    venv.EnvBuilder(with_pip=False).create(path)
    return str(path / "bin" / "python")


def test_ray_supervises_simulators_in_two_isolated_venvs(tmp_path: Path) -> None:
    libero_venv = tmp_path / "libero-venv"
    maniskill_venv = tmp_path / "maniskill-venv"
    libero_python = _create_venv(libero_venv)
    maniskill_python = _create_venv(maniskill_venv)

    with Cluster() as cluster:
        cluster.reserve("simulators", PlacementSpec(bundles_per_node=2))
        libero = cluster.launch(
            "libero",
            SimulatorSupervisor,
            libero_python,
            "libero",
            placement=RolePlacement(pool="simulators", bundle_ranks=(0,)),
        )
        maniskill = cluster.launch(
            "maniskill",
            SimulatorSupervisor,
            maniskill_python,
            "maniskill",
            placement=RolePlacement(pool="simulators", bundle_ranks=(1,)),
        )

        libero_health = libero.call("health").wait()[0]
        maniskill_health = maniskill.call("health").wait()[0]

        assert libero.call("supervisor_executable").wait() == [sys.executable]
        assert maniskill.call("supervisor_executable").wait() == [sys.executable]
        assert libero_health["backend"] == "libero"
        assert maniskill_health["backend"] == "maniskill"
        assert libero_health["protocol_version"] == 1
        assert maniskill_health["protocol_version"] == 1
        assert Path(libero_health["prefix"]).resolve() == libero_venv.resolve()
        assert Path(maniskill_health["prefix"]).resolve() == maniskill_venv.resolve()
        assert libero_health["prefix"] != maniskill_health["prefix"]

        assert libero.call("reset", 10).wait() == [{"observation": {"state": 10}}]
        assert maniskill.call("reset", 20).wait() == [{"observation": {"state": 20}}]
        assert libero.call("step", 3).wait() == [
            {
                "observation": {"state": 13},
                "reward": 13.0,
                "terminated": False,
                "truncated": False,
            }
        ]
        assert maniskill.call("step", -2).wait() == [
            {
                "observation": {"state": 18},
                "reward": 18.0,
                "terminated": False,
                "truncated": False,
            }
        ]

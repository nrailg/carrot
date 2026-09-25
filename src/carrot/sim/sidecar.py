"""Child-interpreter env client. Simulator objects stay in the sidecar process."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from contextlib import suppress
from typing import Any

from carrot.distributed import Worker

PROTOCOL_VERSION = 1

SIDECAR_SERVE_SOURCE = r"""
import json
import sys

def serve(env):
    for line in sys.stdin:
        request = json.loads(line)
        operation = request["op"]
        if operation == "health":
            response = env.health()
        elif operation == "reset":
            response = env.reset(request["seed"])
        elif operation == "step":
            response = env.step(request["action"])
        elif operation == "close":
            env.close()
            print(json.dumps({"closed": True}), flush=True)
            break
        else:
            response = {"error": f"unknown operation: {operation}"}
        print(json.dumps(response), flush=True)
"""


class SimulatorSidecar:
    """Typed env handle for a child Python that owns MuJoCo or a sim plugin."""

    def __init__(
        self,
        python_executable: str,
        program: str,
        *,
        extra_env: Mapping[str, str] | None = None,
    ) -> None:
        environment = os.environ.copy()
        if extra_env:
            environment.update(extra_env)
        self._closed = False
        self._process = subprocess.Popen(
            [python_executable, "-u", "-c", program],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=environment,
        )
        try:
            self.health()
        except Exception:
            self.close()
            raise

    def health(self) -> dict[str, Any]:
        return self._call("health")

    def reset(self, seed: int) -> Any:
        return self._call("reset", seed=seed)

    def step(self, action: Any) -> Any:
        return self._call("step", action=action)

    def close(self) -> None:
        if self._closed:
            return
        if self._process.poll() is None:
            with suppress(RuntimeError):
                self._call("close")
        self._closed = True
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)

    def _call(self, operation: str, **payload: Any) -> Any:
        assert (
            not self._closed
            and self._process.stdin is not None
            and self._process.stdout is not None
        ), (
            "simulator sidecar is not running"
        )
        self._process.stdin.write(json.dumps({"op": operation, **payload}) + "\n")
        self._process.stdin.flush()
        response = self._process.stdout.readline()
        if not response:
            stderr = ""
            if self._process.stderr is not None:
                stderr = self._process.stderr.read()
            raise RuntimeError(f"simulator sidecar exited unexpectedly: {stderr}")
        result = json.loads(response)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result


class SimulatorSupervisor(Worker):
    """Ray worker in the core venv that forwards env calls to one sidecar."""

    def __init__(
        self,
        python_executable: str,
        program: str,
        extra_env: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.python_executable = python_executable
        self.program = program
        self.extra_env = dict(extra_env or {})
        self.sidecar: SimulatorSidecar | None = None

    def setup(self) -> None:
        self.sidecar = SimulatorSidecar(
            self.python_executable,
            self.program,
            extra_env=self.extra_env,
        )

    def health(self) -> dict[str, Any]:
        return self._require_sidecar().health()

    def reset(self, seed: int) -> Any:
        return self._require_sidecar().reset(seed)

    def step(self, action: Any) -> Any:
        return self._require_sidecar().step(action)

    def teardown(self) -> None:
        if self.sidecar is None:
            return
        self.sidecar.close()
        self.sidecar = None

    def _require_sidecar(self) -> SimulatorSidecar:
        assert self.sidecar is not None, "simulator sidecar is not running"
        return self.sidecar

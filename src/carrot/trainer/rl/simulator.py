import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import zlib
from multiprocessing.connection import Client, Connection
from pathlib import Path
from typing import override

from carrot.distributed.worker import Worker


class SimulatorWorker(Worker):
    """Own an isolated simulator subprocess on the GPU assigned by Carrot placement.

    Parameters
    ----------
    script : str
        Explicit Isaac or LIBERO entry point; simulator imports stay out of the Ray actor.
    config : dict
    output : str
    python : str | None
        LIBERO uses its dedicated dependency environment.
    """

    def __init__(
        self, script: str, config: dict, output: str, node_ip: str, python: str | None = None
    ) -> None:
        self.script, self.config, self.output = script, config, Path(output)
        self.python = python or sys.executable
        self.node_ip = node_ip
        self.process: subprocess.Popen | None = None
        self.connection: Connection | None = None
        self.temporary: tempfile.TemporaryDirectory | None = None
        self.log = None

    @override
    def setup(self) -> None:
        assert self.master_addr == self.node_ip, "renderer placed on the wrong node"
        assert Path(self.script).is_file() and Path(self.python).is_file()
        icd = "/etc/vulkan/icd.d/nvidia_icd.json"
        assert Path(icd).is_file(), f"missing explicit renderer ICD: {icd}"
        self.output.mkdir(parents=True, exist_ok=True)
        gpu = self.local_rank
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        physical = int(visible.split(",")[gpu]) if visible else gpu
        name = subprocess.check_output(
            ["nvidia-smi", "-i", str(physical), "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip()
        assert "RTX PRO 5000" in name, f"this experiment requires Pro5000 rendering, got {name}"
        self.temporary = tempfile.TemporaryDirectory(prefix="carrot-paint-")
        socket = str(Path(self.temporary.name) / "sim.sock")
        self.log = (self.output / "simulator.log").open("a", buffering=1)
        try:
            self.process = subprocess.Popen(
                [
                    self.python,
                    self.script,
                    "--socket",
                    socket,
                    "--config",
                    json.dumps(self.config),
                    "--gpu",
                    str(physical),
                ],
                stdout=self.log,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                    "VK_ICD_FILENAMES": icd,
                    "OMP_NUM_THREADS": "4",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "4",
                },
            )
            deadline = time.monotonic() + 300
            while not Path(socket).exists():
                assert self.process.poll() is None, f"simulator startup failed; see {self.output}"
                assert time.monotonic() < deadline, (
                    f"simulator startup timed out; see {self.output}"
                )
                time.sleep(0.1)
            self.connection = Client(socket, family="AF_UNIX")
            assert self.connection.poll(30), f"simulator not ready; see {self.output}"
            assert self.connection.recv() == {"ready": True}
        except BaseException:
            self.teardown()
            raise

    def request(self, method: str, **kwargs: object) -> object:
        """Execute one request, propagating simulator failures and preserving terminal observations.

        Parameters
        ----------
        method : str
        **kwargs : object
            Native numpy payloads over a private local Unix socket; never exposed to the network.

        Returns
        -------
        object
            Simulator result. Collection has its own absolute deadline.
        """
        assert self.connection is not None and self.process is not None
        assert self.process.poll() is None, f"simulator exited; see {self.output}"
        self.connection.send((method, kwargs))
        assert self.connection.poll(24 * 3600 if method == "collect" else 300), (
            f"simulator timeout in {method}; see {self.output}"
        )
        response = self.connection.recv()
        assert response["ok"], response["traceback"]
        return response["result"]

    @override
    def teardown(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self.connection is not None:
            self.connection.close()
        if self.temporary is not None:
            self.temporary.cleanup()
        if self.log is not None:
            self.log.close()

    def files(self, subdirectory: str) -> list[dict]:
        """Describe completed artifacts for verified copying between disjoint filesystems.

        Parameters
        ----------
        subdirectory : str
            Existing directory beneath this worker's persistent output.

        Returns
        -------
        list[dict]
            Relative paths, uncompressed sizes and SHA-256 digests.
        """
        directory = (self.output / subdirectory).resolve()
        assert directory.is_relative_to(self.output.resolve()) and directory.is_dir()
        manifest = []
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                manifest.append(
                    {
                        "path": str(path.relative_to(directory)),
                        "size": path.stat().st_size,
                        "sha256": digest,
                    }
                )
        return manifest

    def read_part(self, subdirectory: str, relative: str, offset: int) -> bytes:
        """Read a bounded, compressed artifact block for cross-pool transfer.

        Parameters
        ----------
        subdirectory, relative : str
            Must resolve beneath the persistent output directory.
        offset : int
            Uncompressed byte offset.

        Returns
        -------
        bytes
            zlib data containing at most 16 MiB of source bytes.
        """
        path = (self.output / subdirectory / relative).resolve()
        assert path.is_relative_to(self.output.resolve()) and offset >= 0
        with path.open("rb") as stream:
            stream.seek(offset)
            return zlib.compress(stream.read(16 * 1024**2), level=1)

    def restart(self, script: str, config: dict, python: str | None = None) -> None:
        """Close one simulator before starting another on the same assigned GPU.

        Parameters
        ----------
        script : str
        config : dict
        python : str | None
            Explicit dependency environment; defaults to the actor's interpreter.
        """
        self.request("close")
        assert self.process is not None
        self.process.wait(timeout=60)
        self.teardown()
        self.script, self.config, self.python = script, config, python or sys.executable
        self.process = self.connection = self.temporary = self.log = None
        self.setup()

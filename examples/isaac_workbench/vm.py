import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from logic import Key
from PIL import Image
from prepare_guest import archive
from tasks import CodingTask


class Qmp:
    """Synchronous QMP transport; asynchronous events are never mistaken for replies."""

    def __init__(self, path: Path, timeout: float = 10) -> None:
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.settimeout(timeout)
        self.socket.connect(str(path))
        self.stream = self.socket.makefile("rb")
        self.sequence = 0
        self.lock = threading.RLock()
        greeting = self._read()
        assert "QMP" in greeting, f"invalid QMP greeting: {greeting}"
        self.execute("qmp_capabilities")

    def _read(self) -> dict:
        line = self.stream.readline()
        assert line, "QMP connection closed before a reply"
        return json.loads(line)

    def execute(self, command: str, arguments: dict | None = None) -> dict:
        with self.lock:
            return self._execute(command, arguments)

    def _execute(self, command: str, arguments: dict | None = None) -> dict:
        self.sequence += 1
        request = {"execute": command, "id": self.sequence}
        if arguments is not None:
            request["arguments"] = arguments
        self.socket.sendall(json.dumps(request).encode() + b"\n")
        while True:
            response = self._read()
            if "event" in response:
                continue
            assert response.get("id") == self.sequence, f"unexpected QMP reply: {response}"
            assert "error" not in response, f"QMP {command} failed: {response}"
            return response["return"]

    def close(self) -> None:
        try:
            self.stream.close()
        finally:
            self.socket.close()


class LinuxVm:
    """Own a fresh, networkless Linux guest. Its only runtime input is physical key edges.

    Parameters
    ----------
    guest_dir : Path
        Contains vmlinuz and initramfs.cpio.gz from prepare_guest.py.
    output : Path
        Dedicated directory for guest console and QEMU diagnostic logs.
    accelerator : str
        Explicitly selected kvm or tcg; failures never silently change backend.
    """

    def __init__(
        self,
        guest_dir: Path,
        output: Path,
        accelerator: str = "kvm",
        task: CodingTask | None = None,
    ) -> None:
        assert accelerator in ("kvm", "tcg")
        assert shutil.which("qemu-system-x86_64"), "install qemu-system-x86 before starting"
        for name in ("vmlinuz", "initramfs.cpio.gz"):
            assert (guest_dir / name).is_file(), f"missing guest artifact: {guest_dir / name}"
        assert not output.exists(), f"VM output must be unique: {output}"
        output.mkdir(parents=True)
        self.episode = uuid.uuid4().hex
        self.owner_pid = os.getpid()
        self.task = task
        self.pending_submission: int | None = None
        self.lock = threading.RLock()
        self.serial_path = output.resolve() / "guest-console.log"
        self.serial_path.write_text("")
        self.log = (output / "qemu.log").open("wb")
        self.runtime = tempfile.TemporaryDirectory(prefix="carrot-vm-")
        self.qmp: Qmp | None = None
        self.process: subprocess.Popen | None = None
        self.down: set[str] = set()
        self.events: list[dict] = []
        runtime = Path(self.runtime.name)
        initrd = guest_dir.resolve() / "initramfs.cpio.gz"
        if task is not None:
            seed = runtime / "seed"
            seed.mkdir()
            (seed / "task.json").write_text(json.dumps(task.guest_spec(self.episode)))
            overlay = runtime / "task.cpio.gz"
            archive(seed, overlay)
            initrd = runtime / "initramfs.cpio.gz"
            with initrd.open("wb") as target:
                for part in (guest_dir / "initramfs.cpio.gz", overlay):
                    with part.open("rb") as source:
                        shutil.copyfileobj(source, target)
        self.frame_path = runtime / "screen.ppm"
        socket_path = runtime / "qmp.sock"
        command = [
            "qemu-system-x86_64",
            "-accel",
            accelerator,
            "-m",
            "512",
            "-smp",
            "1",
            "-kernel",
            str(guest_dir.resolve() / "vmlinuz"),
            "-initrd",
            str(initrd),
            "-append",
            "console=ttyS0 console=tty1 rdinit=/init quiet panic=-1",
            "-display",
            "none",
            "-vga",
            "std",
            "-nic",
            "none",
            "-no-reboot",
            "-qmp",
            f"unix:{socket_path},server=on,wait=off",
            "-serial",
            f"file:{self.serial_path}",
        ]
        if accelerator == "kvm":
            command += ["-cpu", "host"]
        try:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name("qemu_child.py")),
                    str(self.owner_pid),
                    *command,
                ],
                stdout=self.log,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + 90
            while not socket_path.exists():
                self._check_alive(deadline)
                time.sleep(0.05)
            self.qmp = Qmp(socket_path)
            while "WORKBENCH_READY" not in self.serial_path.read_text(errors="replace"):
                self._check_alive(deadline)
                time.sleep(0.1)
            self.qmp.execute("stop")
        except BaseException:
            self.close()
            raise

    def _check_alive(self, deadline: float) -> None:
        assert self.process is not None
        assert self.process.poll() is None, f"QEMU exited; inspect {self.log.name}"
        assert time.monotonic() < deadline, f"Linux boot timed out; inspect {self.serial_path}"

    def key(self, qcode: str, down: bool, physics_step: int) -> None:
        with self.lock:
            self._key(qcode, down, physics_step)

    def _key(self, qcode: str, down: bool, physics_step: int) -> None:
        assert os.getpid() == self.owner_pid, "create each VM inside its owning spawned worker"
        assert self.qmp is not None
        assert (qcode in self.down) != down, f"duplicate HID edge: {qcode} down={down}"
        if qcode == "ret" and down and self.task is not None:
            self.pending_submission = self.grade()["attempt"] + 1
        # QEMU rejects input-send-event while stopped. Run only within this locked edge window.
        self.qmp.execute("cont")
        try:
            self.qmp.execute(
                "input-send-event",
                {
                    "events": [
                        {
                            "type": "key",
                            "data": {"down": down, "key": {"type": "qcode", "data": qcode}},
                        }
                    ]
                },
            )
            time.sleep(0.025)
        finally:
            self.qmp.execute("stop")
        if down:
            self.down.add(qcode)
        else:
            self.down.remove(qcode)
        self.events.append({"physics_step": physics_step, "qcode": qcode, "down": down})
        if qcode == "ret" and not down and self.pending_submission is not None:
            deadline = time.monotonic() + 5
            while self.grade()["attempt"] < self.pending_submission:
                assert time.monotonic() < deadline, (
                    "guest did not finish grading the submitted expression"
                )
                self.advance(0.025)
            self.pending_submission = None

    def advance(self, seconds: float) -> None:
        with self.lock:
            assert self.qmp is not None and 0 < seconds <= 1
            self.qmp.execute("cont")
            time.sleep(seconds)
            self.qmp.execute("stop")

    def screenshot(self) -> Image.Image:
        with self.lock:
            assert self.qmp is not None
            self.qmp.execute("screendump", {"filename": str(self.frame_path)})
            with Image.open(self.frame_path) as frame:
                return frame.convert("RGB")

    def grade(self) -> dict:
        assert self.process is not None and self.process.poll() is None, "guest process exited"
        records = [
            json.loads(line.removeprefix("WORKBENCH_GRADE "))
            for line in self.serial_path.read_text(errors="replace").splitlines()
            if line.startswith("WORKBENCH_GRADE ")
        ]
        if not records:
            return {
                "episode": self.episode,
                "attempt": 0,
                "passed": 0,
                "total": 1,
                "success": False,
            }
        assert all(record["episode"] == self.episode for record in records), "stale guest grade"
        return records[-1]

    def close(self) -> None:
        assert os.getpid() == self.owner_pid, "only the creating worker may close its VM"
        try:
            if self.qmp is not None:
                self.qmp.close()
                self.qmp = None
        finally:
            try:
                if self.process is not None and self.process.poll() is None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=5)
            finally:
                self.log.close()
                self.runtime.cleanup()


class VmPool:
    """One VM per environment, with local resets and barrier-synchronized I/O.

    Each VM owns a temporary Unix socket, RAM filesystem, serial log and episode nonce.
    Only immutable kernel/base initramfs files are shared. No TCP port is allocated.
    Calls are serialized across step/reset/close; a failed VM operation fails the whole batch.
    """

    def __init__(
        self, guest_dir: Path, output: Path, tasks: list[CodingTask], accelerator: str = "kvm"
    ) -> None:
        assert tasks and not output.exists()
        self.guest_dir, self.output, self.tasks = guest_dir, output, tasks
        self.accelerator = accelerator
        self.owner_pid = os.getpid()
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=min(8, len(tasks)))
        self.episodes = [-1] * len(tasks)
        self.guests: list[LinuxVm | None] = [None] * len(tasks)
        self.pressed = [set() for _ in tasks]
        try:
            self.reset(list(range(len(tasks))))
        except BaseException:
            self.close()
            raise

    def _barrier(self, jobs: list[Callable]) -> list:
        assert os.getpid() == self.owner_pid, "construct VmPool after spawning each RL worker"
        futures = [self.executor.submit(job) for job in jobs]
        values, errors = [], []
        for future in futures:
            try:
                values.append(future.result())
            except BaseException as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("VM batch failed", errors)
        return values

    def reset(self, indices: list[int]) -> None:
        with self.lock:
            assert len(set(indices)) == len(indices)
            assert all(0 <= i < len(self.guests) for i in indices)

            def reset_one(i: int) -> None:
                if self.guests[i] is not None:
                    self.guests[i].close()
                    self.guests[i] = None
                self.episodes[i] += 1
                self.pressed[i].clear()
                output = self.output / f"env-{i:03d}" / f"episode-{self.episodes[i]:04d}"
                self.guests[i] = LinuxVm(self.guest_dir, output, self.accelerator, self.tasks[i])

            self._barrier([lambda i=i: reset_one(i) for i in indices])

    def step(self, batches: list[list[tuple[int, int, bool]]], keys: list[Key]) -> list[dict]:
        with self.lock:
            assert len(batches) == len(self.guests)

            def update(i: int) -> dict:
                guest = self.guests[i]
                assert guest is not None
                for tick, index, down in batches[i]:
                    assert (index in self.pressed[i]) != down, "duplicate physical key edge"
                    if down:
                        self.pressed[i].add(index)
                    else:
                        self.pressed[i].remove(index)
                    desired = {
                        code
                        for k in self.pressed[i]
                        for code in (*keys[k].modifiers, keys[k].qcode)
                    }
                    # Release character keys before modifiers; press modifiers before characters.
                    for code in sorted(guest.down - desired, key=lambda c: c in ("shift", "ctrl")):
                        guest.key(code, False, tick)
                    for code in sorted(
                        desired - guest.down, key=lambda c: c not in ("shift", "ctrl")
                    ):
                        guest.key(code, True, tick)
                return guest.grade()

            return self._barrier([lambda i=i: update(i) for i in range(len(self.guests))])

    def screens(self) -> list[Image.Image]:
        with self.lock:
            return self._barrier([lambda guest=guest: guest.screenshot() for guest in self.guests])

    def close(self) -> None:
        assert os.getpid() == self.owner_pid, "only the creating worker may close its VM pool"
        with self.lock:
            errors = []
            for guest in self.guests:
                if guest is not None:
                    try:
                        guest.close()
                    except Exception as error:
                        errors.append(error)
            self.executor.shutdown(wait=True, cancel_futures=True)
            if errors:
                raise ExceptionGroup("VM cleanup failed", errors)

import argparse
import gzip
import hashlib
import json
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path


def dependencies(binary: Path) -> list[Path]:
    result = subprocess.run(["ldd", str(binary)], check=True, capture_output=True, text=True)
    assert "not found" not in result.stdout, result.stdout
    return [Path(p) for p in re.findall(r"(?:=>\s+|^\s*)(/\S+)", result.stdout, re.MULTILINE)]


def archive(root: Path, destination: Path) -> None:
    with gzip.open(destination, "wb") as output:
        paths = [root, *sorted(root.rglob("*"))]
        for inode, path in enumerate(paths + [None], start=1):
            if path is None:
                name, mode, data = "TRAILER!!!", 0, b""
            else:
                name = "." if path == root else str(path.relative_to(root))
                mode = path.lstat().st_mode
                data = (
                    str(path.readlink()).encode()
                    if path.is_symlink()
                    else (path.read_bytes() if path.is_file() else b"")
                )
            encoded = name.encode() + b"\0"
            fields = (inode, mode, 0, 0, 1, 0, len(data), 0, 0, 0, 0, len(encoded), 0)
            header = b"070701" + "".join(f"{v:08x}" for v in fields).encode()
            output.write(header + encoded + b"\0" * (-(110 + len(encoded)) % 4))
            output.write(data + b"\0" * (-len(data) % 4))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a real Linux/Python initramfs from local files"
    )
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument(
        "--kernel-root", type=Path, required=True, help="extracted kernel/modules root"
    )
    parser.add_argument("--kernel-version", required=True)
    parser.add_argument("--python", type=Path, default=Path("/usr/bin/python3.10"))
    parser.add_argument("--stdlib", type=Path, default=Path("/usr/lib/python3.10"))
    parser.add_argument("--busybox", type=Path, default=Path("/bin/busybox"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.kernel.is_file() and args.python.is_file() and args.busybox.is_file()
    assert not args.output.exists(), f"refusing to replace existing guest: {args.output}"
    args.output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="carrot-initramfs-") as temporary:
        root = Path(temporary)
        for directory in ("bin", "dev", "proc", "sys", "tmp", "work", "usr/bin", "lib/modules"):
            (root / directory).mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.busybox, root / "bin/busybox")
        for applet in ("sh", "mount", "mkdir", "insmod", "sleep", "poweroff"):
            (root / "bin" / applet).symlink_to("busybox")
        shutil.copytree(
            args.stdlib,
            root / str(args.stdlib).lstrip("/"),
            ignore=shutil.ignore_patterns("__pycache__", "test", "tests"),
        )
        shutil.copy2(args.python, root / "usr/bin/python3")
        libraries = set(dependencies(args.python))
        for extension in args.stdlib.rglob("*.so"):
            libraries.update(dependencies(extension))
        for library in libraries:
            target = root / str(library).lstrip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(library, target)
        config = (args.kernel_root / "boot" / f"config-{args.kernel_version}").read_text()
        for feature in ("DEVTMPFS", "KEYBOARD_ATKBD", "SERIO_I8042", "VT", "VGA_CONSOLE"):
            assert f"CONFIG_{feature}=y\n" in config, f"guest kernel needs built-in {feature}"
        shutil.copy2(Path(__file__).with_name("guest_console.py"), root / "guest_console.py")
        shutil.copy2(Path(__file__).with_name("guest_task.py"), root / "guest_task.py")
        init = "\n".join(
            [
                "#!/bin/sh",
                "set -eu",
                "mount -t devtmpfs devtmpfs /dev",
                "mount -t proc proc /proc",
                "mount -t sysfs sysfs /sys",
                "cd /work",
                "export TERM=linux HOME=/work",
                "if [ -f /task.json ]; then",
                "  exec /usr/bin/python3 -u /guest_task.py </dev/tty1 >/dev/tty1 2>&1",
                "fi",
                "exec /usr/bin/python3 -u /guest_console.py </dev/tty1 >/dev/tty1 2>&1",
                "",
            ]
        )
        (root / "init").write_text(init)
        (root / "init").chmod(stat.S_IFREG | 0o755)
        archive(root, args.output / "initramfs.cpio.gz")
    shutil.copy2(args.kernel, args.output / "vmlinuz")
    manifest = {
        name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
        for name in ("vmlinuz", "initramfs.cpio.gz")
    }
    manifest.update(
        kernel_version=args.kernel_version, python=str(args.python), stdlib=str(args.stdlib)
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

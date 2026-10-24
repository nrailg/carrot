"""One-shot dependency: wait for collection, verify and publish its H20 copy."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    status = args.source / "task-status.json"
    print("waiting for collection task-status.json", flush=True)
    while not status.exists():
        time.sleep(30)
    assert json.loads(status.read_text())["complete"], "collection failed; do not train"
    marker = json.loads((args.source / "manifest.json").read_text())
    gate = json.loads((args.source / "teacher-gate.json").read_text())
    assert gate["passed"] is True and len(gate["episodes"]) == 16
    assert not args.destination.exists()
    staging = args.destination.with_name(args.destination.name + ".partial")
    staging.mkdir(parents=True, exist_ok=False)
    entries = marker["files"]
    expected = {entry["path"] for entry in entries}
    actual = {
        str(path.relative_to(args.source))
        for path in (args.source / "demos").rglob("*")
        if path.is_file()
    }
    assert expected == actual and len(expected) == len(entries)
    for index, entry in enumerate(entries):
        source = (args.source / entry["path"]).resolve()
        assert source.is_relative_to((args.source / "demos").resolve())
        target = staging / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        with target.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        assert target.stat().st_size == entry["size"] and digest == entry["sha256"], target
        if index % 50 == 0:
            print(json.dumps({"verified_files": index + 1, "total": len(entries)}), flush=True)
    for name in ("manifest.json", "teacher-gate.json", "camera.json", "task-status.json"):
        shutil.copy2(args.source / name, staging / name)
    staging.rename(args.destination)
    print(json.dumps({"published": str(args.destination), "files": len(entries)}), flush=True)


if __name__ == "__main__":
    main()

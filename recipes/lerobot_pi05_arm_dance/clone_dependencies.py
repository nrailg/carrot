import argparse
import importlib.metadata as metadata
import json
import shutil
from collections import deque
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {}
    for root in [args.base, args.target]:
        for dist in metadata.distributions(path=[str(root)]):
            sources[canonicalize_name(dist.metadata["Name"])] = (root, dist)
    queue = deque([Requirement("lerobot[pi,training]==0.6.1")])
    visited = set()
    selected = {}
    while queue:
        req = queue.popleft()
        name = canonicalize_name(req.name)
        key = (name, tuple(sorted(req.extras)))
        if key in visited:
            continue
        visited.add(key)
        assert name in sources, f"Missing offline dependency: {req}"
        root, dist = sources[name]
        assert dist.version in req.specifier, f"{req}: found {dist.version} at {root}"
        selected[name] = (root, dist)
        for line in dist.requires or []:
            dependency = Requirement(line)
            if dependency.marker is None or any(
                dependency.marker.evaluate({"extra": extra}) for extra in {"", *req.extras}
            ):
                queue.append(dependency)

    manifest = []
    for name, (root, dist) in sorted(selected.items()):
        if root == args.base:
            assert dist.files is not None, f"Missing RECORD: {name}"
            for relative in dist.files:
                # Console scripts belong to the original venv; use module entrypoints instead.
                if ".." in relative.parts or "__pycache__" in relative.parts:
                    continue
                source = root / relative
                assert source.is_file(), f"Missing dependency file: {source}"
                destination = args.target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        manifest.append({"name": name, "version": dist.version, "source": str(root)})
        print(f"{name}=={dist.version}", flush=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

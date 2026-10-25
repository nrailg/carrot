"""One-shot dependency: wait for collection, verify and publish its H20 copy."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path


def audit_episodes(source: Path) -> dict:
    """Reject duplicate seeds or complete action/state sequences before publication.

    Parameters
    ----------
    source : Path
        Completed collection containing exactly 64 successful training episodes.

    Returns
    -------
    dict
        Episode identities and SHA256 fingerprints for the published audit record.
    """
    episodes = sorted((source / "demos").glob("episode-*"))
    assert len(episodes) == 64, f"expected 64 episodes, found {len(episodes)}"
    rows = []
    for episode in episodes:
        info = json.loads((episode / "result.json").read_text())
        assert info["success"] and info["picks"] == info["returns"] == 3 and info["drops"] == 0
        assert 200000 <= info["seed"] < 200128, f"non-training seed: {episode}"
        row = {"episode": episode.name, "seed": info["seed"]}
        for name in ("actions", "state"):
            with (episode / f"{name}.npy").open("rb") as stream:
                row[name + "_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        rows.append(row)
    for key in ("seed", "actions_sha256", "state_sha256"):
        seen = {}
        for row in rows:
            value = row[key]
            assert value not in seen, f"duplicate {key}: {seen.get(value)} and {row['episode']}"
            seen[value] = row["episode"]
    return {"passed": True, "episodes": rows, "unique_seeds": 64, "unique_trajectories": 64}


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
    assert gate["task_contract"] == "static-reference-v1"
    audit = audit_episodes(args.source)
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
    (staging / "deduplication.json").write_text(json.dumps(audit, indent=2) + "\n")
    staging.rename(args.destination)
    print(json.dumps({"published": str(args.destination), "files": len(entries)}), flush=True)


if __name__ == "__main__":
    main()

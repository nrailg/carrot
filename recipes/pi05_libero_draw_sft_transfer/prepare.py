import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import yaml

from carrot.data.libero import build_dataset
from carrot.data.libero_draw import DrawDataset, ScheduledDataset
from carrot.data.mixture_schedule import make_schedule, valid_lengths


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-gate", type=Path, required=True)
    args = parser.parse_args()
    dfs = Path(os.environ["MY_DFS"])
    source = Path(__file__).resolve().parents[2]
    gate = json.loads(args.teacher_gate.read_text())
    assert gate["teacher_mode"] == "events" and gate["passed"] is True
    assert gate["task_contract"] == "static-reference-v1", "old oracle-guided task"
    assert sorted(row["seed"] for row in gate["episodes"]) == list(range(300000, 300016))
    assert all(row["success"] and row["picks"] == row["returns"] == 3 for row in gate["episodes"])
    assert gate["carrot_sha256"] == digest(source / "examples/isaac_workbench/multicolor_scene.py")
    assert not args.output.exists(), f"existing experiment: {args.output}"
    draw = DrawDataset(str(args.draw_root))
    assert len(draw.episodes) == 64 and all(200000 <= s < 200128 for s in draw.seeds)
    marker = json.loads((args.draw_root.parent / "manifest.json").read_text())
    names = [entry["path"] for entry in marker["files"]]
    actual = {
        str(path.relative_to(args.draw_root.parent))
        for path in args.draw_root.rglob("*")
        if path.is_file()
    }
    assert names and len(names) == len(set(names)) and set(names) == actual
    for entry in marker["files"]:
        path = (args.draw_root.parent / entry["path"]).resolve()
        assert path.is_relative_to(args.draw_root.parent.resolve()) and path.is_file()
        assert path.stat().st_size == entry["size"] and digest(path) == entry["sha256"], path
    root = dfs / "hf-hub/lerobot/libero"
    libero = build_dataset(root=str(root)).dataset
    ids = np.asarray(libero.source.hf_dataset["episode_index"]).reshape(-1)
    lengths = valid_lengths(ids)
    assert len(lengths) == len(libero)
    args.output.mkdir(parents=True)
    config = yaml.safe_load((Path(__file__).with_name("train.yaml")).read_text())
    base = dfs / "experiments/carrot/pi05_libero_sft_2k_20260919/init"
    tokenizer = dfs / "hf-hub/google/paligemma-3b-pt-224"
    norm = (
        dfs
        / "hf-hub/Physical-Intelligence/pi05_libero_pytorch"
        / "assets/physical-intelligence/libero/norm_stats.json"
    )
    assert (base / "model.safetensors").is_file() and (tokenizer / "tokenizer.json").is_file()
    assert norm.is_file()
    provenance = {
        "carrot_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip()
        if (source / ".git").exists()
        else os.environ["CARROT_COMMIT"],
        "base_sha256": digest(base / "model.safetensors"),
        "base_config_sha256": digest(base / "config.json"),
        "norm_sha256": digest(norm),
        "teacher_gate": gate,
        "draw_manifest_sha256": digest(args.draw_root.parent / "manifest.json"),
        "training_seed": 1000,
        "draw_seeds": draw.seeds,
        "libero_frames": len(libero),
    }
    plans = {}
    for name, mixed in (("libero", False), ("mixed", True)):
        plan = make_schedule(lengths, draw.lengths, mixed=mixed)
        manifest = args.output / f"{name}-samples.npz"
        np.savez(manifest, **plan)
        plans[name] = plan
        dataset = ScheduledDataset(libero, draw, str(manifest))
        for index in (0, 8, 255, len(dataset) - 1):
            dataset[index]
        config["model"] = {"path": str(base), "tokenizer_path": str(tokenizer)}
        config["dataset"].update(
            factory_kwargs={
                "root": str(root),
                "draw_root": str(args.draw_root),
                "manifest": str(manifest),
            },
            norm_stats_path=str(norm),
        )
        config["output_dir"] = str(args.output / name)
        config["wandb"]["name"] = args.output.name + "-" + name
        (args.output / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        provenance[name] = {
            "manifest_sha256": digest(manifest),
            "samples": {str(i): int((plan["domain"] == i).sum()) for i in range(2)},
            "valid_steps": {
                str(i): int(plan["valid_length"][plan["domain"] == i].sum()) for i in range(2)
            },
        }
    assert np.array_equal(
        plans["libero"]["source_index"][:25600],
        plans["mixed"]["source_index"][plans["mixed"]["domain"] == 0],
    )
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"prepared": str(args.output), "routes": ["libero", "mixed"]}), flush=True)


if __name__ == "__main__":
    main()

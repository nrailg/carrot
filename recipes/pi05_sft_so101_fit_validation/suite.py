"""Run and archive the approved SO101 fit-reduction sequence."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from recipes.pi05_sft_so101_fit_validation.fit import fixed_noise

CARROT_DIR = Path(__file__).resolve().parents[2]
RECIPE_DIR = Path(__file__).resolve().parent
MY_DFS = Path(os.environ["MY_DFS"])
DATASET_ROOT = MY_DFS / "hf-hub/nrailg/knock_down_the_cylinder_1_20260930_222251"
BASE_RECIPE = CARROT_DIR / "recipes/pi05_sft_so101_knock_down_the_cylinder_overfit/train.yaml"
BASE_MODEL = MY_DFS / "hf-hub/Physical-Intelligence/pi05_base_pytorch"
TOKENIZER = MY_DFS / "hf-hub/google/paligemma-3b-pt-224"


def _make_config(case: dict, output: Path, base: dict, settings: dict) -> dict:
    """Build one cumulative ablation from the same model and data contract."""
    config = json.loads(json.dumps(base))
    config["model"]["path"] = str(BASE_MODEL)
    config["model"]["tokenizer_path"] = str(TOKENIZER)
    config["dataset"]["factory"] = "recipes.pi05_sft_so101_fit_validation.fit.build_dataset"
    kwargs = config["dataset"]["factory_kwargs"]
    kwargs.update({
        "frame_index": case["frame_index"],
        "repeat_count": 64,
        "repo_id": "nrailg/knock_down_the_cylinder_1_20260930_222251",
        "revision": None,
        "root": str(DATASET_ROOT),
        "action_horizon": settings["action_horizon"],
        "base_image_key": None,
        "wrist_image_key": "observation.images.wrist",
    })
    config["optimizer"].update({
        "learning_rate": settings["learning_rate"],
        "warmup_steps": settings["warmup_steps"],
        "decay_steps": settings["steps"],
        "decay_learning_rate": settings["learning_rate"],
    })
    config["steps"] = settings["steps"]
    config["save_freq"] = settings["steps"]
    config["output_dir"] = str(output)
    config["seed"] = 1000
    config["fit"] = {"vision": case["vision"], "noise_seed": case["noise_seed"]}
    return config


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _train(config_path: Path, case_dir: Path, interval: int, steps: int) -> None:
    """Run training and emit progress at the user's requested interval."""
    log_path = case_dir / "driver.log"
    last_report = time.monotonic()
    with log_path.open("x") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", str(RECIPE_DIR / "train.py"), "--config", str(config_path)],
            cwd=CARROT_DIR, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(),
        )
        while process.poll() is None:
            time.sleep(30)
            if time.monotonic() - last_report >= interval:
                print(json.dumps({"stage": "training", "case": case_dir.name,
                                  "elapsed_seconds": int(time.monotonic() - last_report),
                                  "driver_pid": process.pid}), flush=True)
                last_report = time.monotonic()
        log.flush()
    assert process.returncode == 0, f"training failed for {case_dir.name}; see {log_path}"
    output = log_path.read_text()
    matches = re.findall(r"SFT finished: (\{[^\n]+\})", output)
    assert matches, f"missing SFT completion marker for {case_dir.name}"
    result = ast.literal_eval(matches[-1])
    assert result["step"] == steps, result
    rows = [(int(s), float(loss), float(lr), float(g)) for s, loss, lr, g in re.findall(
        r"step=(\d+) loss=([\d.e+-]+) lr=([\d.e+-]+) grad_norm=([\d.e+-]+)", output
    )]
    assert [row[0] for row in rows] == list(range(1, steps + 1))
    assert all(math.isfinite(value) for row in rows for value in row)
    assert all(row[2] == 1e-6 for row in rows if row[0] >= 100)
    assert "FIT_PARAMETER_UPDATE" in output
    _write_json(case_dir / "training_validation.json", {
        "exit_code": process.returncode, "step": result["step"], "final_loss": result["loss"],
        "first100_mean_loss": float(np.mean([r[1] for r in rows[:100]])),
        "last100_mean_loss": float(np.mean([r[1] for r in rows[-100:]])),
    })
    print(json.dumps({"stage": "training_complete", "case": case_dir.name,
                      "step": result["step"], "loss": result["loss"]}), flush=True)


def _evaluate(config_path: Path, checkpoint: Path, output: Path, nfe: int = 10) -> None:
    subprocess.run(
        [sys.executable, "-u", str(RECIPE_DIR / "evaluate.py"), "--config", str(config_path),
         "--checkpoint", str(checkpoint), "--output", str(output), "--num-steps", str(nfe)],
        cwd=CARROT_DIR, env=os.environ.copy(), check=True,
    )


def _verify_predictions(case_dir: Path, case: dict, horizon: int) -> None:
    table = pa.concat_tables([pq.read_table(p, columns=["index", "episode_index", "action"])
                              for p in sorted((DATASET_ROOT / "data").rglob("*.parquet"))])
    indices = np.asarray(table["index"])
    assert np.array_equal(indices, np.arange(264))
    raw_actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    episodes = np.asarray(table["episode_index"])
    assert np.all(episodes == 0)
    verified_metrics = {}
    for label in ["base", "step1000"] + (
        ["step1000_nfe50"] if case["noise_seed"] is not None else []
    ):
        folder = case_dir / "evaluation" / label
        metrics = json.loads((folder / "metrics.json").read_text())
        files = list(folder.glob("frame_*.npz"))
        assert len(files) == metrics["samples"] == (292 if case["frame_index"] is None else 8)
        errors = []
        primary_errors = []
        probe_errors = []
        primary_frames = []
        for row in metrics["per_sample"]:
            frame, seed = row["frame"], row["noise_seed"]
            name = f"frame_{frame:06d}_noise_{seed}.npz"
            with np.load(folder / name) as sample:
                valid_count = min(horizon, 264 - frame)
                expected = raw_actions[np.minimum(np.arange(frame, frame + horizon), 263)]
                np.testing.assert_array_equal(sample["reference"], expected)
                np.testing.assert_array_equal(sample["valid"], np.arange(horizon) < valid_count)
                np.testing.assert_array_equal(sample["noise"], fixed_noise(seed, horizon, 32))
                assert np.isfinite(sample["predicted"]).all()
                error = sample["predicted"][sample["valid"]] - expected[sample["valid"]]
                np.testing.assert_allclose(np.abs(error).mean(axis=0), row["joint_mae"], rtol=1e-6)
                errors.append(error)
                is_primary = (
                    seed == 1000 + frame if case["frame_index"] is None
                    else seed == case["noise_seed"] if case["noise_seed"] is not None
                    else True
                )
                if is_primary:
                    primary_errors.append(error)
                    primary_frames.append(frame)
                else:
                    probe_errors.append(error)
        error = np.concatenate(errors)
        assert len(error) == metrics["valid_action_rows"]
        for key, actual in [("joint_mae", np.abs(error).mean(axis=0)),
                            ("joint_p95", np.quantile(np.abs(error), 0.95, axis=0)),
                            ("joint_max", np.abs(error).max(axis=0))]:
            np.testing.assert_allclose(actual, metrics[key], rtol=1e-6)
        if case["frame_index"] is None:
            assert sorted(primary_frames) == list(range(264))
        primary = np.concatenate(primary_errors)
        verified_metrics[label] = {
            "primary_samples": len(primary_errors), "primary_valid_action_rows": len(primary),
            "primary_joint_mae": np.abs(primary).mean(axis=0).tolist(),
            "primary_joint_p95": np.quantile(np.abs(primary), 0.95, axis=0).tolist(),
            "primary_joint_max": np.abs(primary).max(axis=0).tolist(),
            "probe_samples": len(probe_errors),
            "probe_joint_mae": np.abs(np.concatenate(probe_errors)).mean(axis=0).tolist()
                if probe_errors else None,
        }
    _write_json(case_dir / "prediction_validation.json", {
        "raw_source_reference": "exact match", "noise": "exact match",
        "mae_p95_max": "independently recomputed", "source_frames": 264,
        "metrics": verified_metrics,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-root", type=Path)
    args = parser.parse_args()
    assert os.environ.get("RAY_ADDRESS"), "set RAY_ADDRESS to the confirmed cluster"
    assert DATASET_ROOT.joinpath("meta/info.json").is_file(), DATASET_ROOT
    assert BASE_MODEL.joinpath("model.safetensors").is_file(), BASE_MODEL
    assert TOKENIZER.joinpath("tokenizer.json").is_file(), TOKENIZER
    base = yaml.safe_load(BASE_RECIPE.read_text())
    settings = yaml.safe_load((RECIPE_DIR / "cases.yaml").read_text())
    assert settings["action_horizon"] == 10 and settings["steps"] == 1000
    start = datetime.now(UTC)
    output_root = (
        MY_DFS / "experiments/carrot/pi05_so101_fit_validation" / start.strftime("%Y%m%dT%H%M%SZ")
    )
    if args.resume_root is not None:
        output_root = args.resume_root.resolve()
        assert output_root.parent == MY_DFS / "experiments/carrot/pi05_so101_fit_validation"
        previous = json.loads((output_root / "suite.json").read_text())
        assert previous["cases"] == settings["cases"]
        assert not (output_root / "completion.json").exists()
    else:
        output_root.mkdir(parents=True, exist_ok=False)
    suffix = start.strftime("%Y%m%dT%H%M%SZ")
    snapshot = output_root / (
        f"resume_source_snapshot_{suffix}" if args.resume_root else "source_snapshot"
    )
    snapshot.mkdir()
    source_paths = list((CARROT_DIR / "src/carrot").rglob("*.py")) + list(RECIPE_DIR.glob("*.py"))
    source_paths += [RECIPE_DIR / name for name in ["cases.yaml", "run.sh"]]
    hashes = {str(path.relative_to(CARROT_DIR)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in source_paths}
    for path in RECIPE_DIR.iterdir():
        if path.is_file():
            (snapshot / path.name).write_bytes(path.read_bytes())
    _write_json(output_root / (
        f"resume_environment_{suffix}.json" if args.resume_root else "environment.json"
    ), {
        "carrot_commit": "d4999bbe6557c2b91273dbe3abbd229532882d66 plus uncommitted changes",
        "docker_image_tag": "not recorded", "upstream_source_commits": "not recorded",
        "python": sys.version, "versions": {name: version(name) for name in
            ["torch", "lerobot", "transformers", "ray", "numpy", "pyarrow"]},
        "source_hashes": hashes, "independent_rerun_requested": True,
    })
    _write_json(output_root / (
        f"resume_suite_{suffix}.json" if args.resume_root else "suite.json"
    ), {
        "started_utc": start.isoformat(), "output_root": str(output_root),
        "dataset_root": str(DATASET_ROOT), "dataset_frames": 264,
        "base_model": str(BASE_MODEL), "ray_address": os.environ["RAY_ADDRESS"],
        "cases": settings["cases"],
    })
    for case in settings["cases"]:
        case_dir = output_root / case["name"]
        config = _make_config(case, case_dir / "training_output", base, settings)
        config_path = case_dir / "train.yaml"
        if case_dir.exists():
            assert args.resume_root is not None
            assert yaml.safe_load(config_path.read_text()) == config
            validation = json.loads((case_dir / "training_validation.json").read_text())
            assert validation["exit_code"] == 0 and validation["step"] == settings["steps"]
            print(json.dumps({"stage": "reuse_completed_training", "case": case["name"]}),
                  flush=True)
        else:
            case_dir.mkdir()
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            _write_json(
                case_dir / "case.json", case | {"action_horizon": settings["action_horizon"]}
            )
            _train(config_path, case_dir, settings["monitor_seconds"], settings["steps"])
        checkpoint = Path(config["output_dir"]) / f"checkpoints/step-{settings['steps']:08d}"
        assert (checkpoint / "model.safetensors").is_file(), checkpoint
        evaluations = [("base", BASE_MODEL, 10), ("step1000", checkpoint, 10)]
        if case["noise_seed"] is not None:
            evaluations.append(("step1000_nfe50", checkpoint, 50))
        for label, model_path, nfe in evaluations:
            eval_dir = case_dir / "evaluation" / label
            if eval_dir.exists():
                assert args.resume_root is not None and (eval_dir / "metrics.json").is_file()
            else:
                _evaluate(config_path, model_path, eval_dir, nfe=nfe)
        _verify_predictions(case_dir, case, settings["action_horizon"])
        base_metrics = json.loads((case_dir / "evaluation/base/metrics.json").read_text())
        fit_metrics = json.loads((case_dir / "evaluation/step1000/metrics.json").read_text())
        verified = json.loads((case_dir / "prediction_validation.json").read_text())
        primary_metrics = verified["metrics"]["step1000"]
        assert base_metrics["horizon"] == fit_metrics["horizon"] == settings["action_horizon"]
        assert base_metrics["unique_frames"] == fit_metrics["unique_frames"]
        result = {
            "case": case,
            "base_range_normalized_rmse": base_metrics["range_normalized_rmse"],
            "step1000_range_normalized_rmse": fit_metrics["range_normalized_rmse"],
            "range_normalized_rmse_reduction_percent": (
                1 - fit_metrics["range_normalized_rmse"] /
                base_metrics["range_normalized_rmse"]
            ) * 100,
            "step1000_joint_mae": fit_metrics["joint_mae"],
            "step1000_joint_p95": fit_metrics["joint_p95"],
            "step1000_joint_max": fit_metrics["joint_max"],
            "valid_action_rows": fit_metrics["valid_action_rows"],
            "primary_joint_mae": primary_metrics["primary_joint_mae"],
            "primary_joint_p95": primary_metrics["primary_joint_p95"],
            "primary_joint_max": primary_metrics["primary_joint_max"],
            "primary_samples": primary_metrics["primary_samples"],
            "primary_scope": "same frame and training noise" if case["noise_seed"] is not None
                else "264 unique frames once each" if case["frame_index"] is None
                else "one training frame with eight noise seeds",
        }
        _write_json(case_dir / "result.json", result)
        print(json.dumps({"stage": "case_complete", "case": case["name"],
                          "checkpoint": str(checkpoint),
                          "range_normalized_rmse_reduction_percent": result[
                              "range_normalized_rmse_reduction_percent"]}), flush=True)
    _write_json(output_root / "completion.json", {
        "completed_utc": datetime.now(UTC).isoformat(),
        "cases": [case["name"] for case in settings["cases"]],
        "results": [json.loads((output_root / case["name"] / "result.json").read_text())
                    for case in settings["cases"]],
    })
    print("FIT_VALIDATION_SUITE_COMPLETE", output_root, flush=True)


if __name__ == "__main__":
    main()

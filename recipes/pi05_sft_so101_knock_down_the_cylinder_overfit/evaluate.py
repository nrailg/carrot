"""Compare fixed training observations before and after the small-data SFT run."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from carrot.data.so101 import build_dataset
from carrot.models.pi05.inference import create_so101_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dataset_config = yaml.safe_load(Path(__file__).with_name("train.yaml").read_text())[
        "dataset"
    ]["factory_kwargs"]
    spec = build_dataset(**(dataset_config | {"root": str(args.dataset_root)}))
    stats = {"state": spec.state_stats, "actions": spec.action_stats}
    stats_path = args.output / "norm_stats.json"
    stats_path.write_text(json.dumps({
        key: {name: np.asarray(value).tolist() for name, value in values.items()}
        for key, values in stats.items()
    } | {"joint_units": "radians", "gripper_units": "fraction"}))
    policy = create_so101_policy(
        args.checkpoint, device=args.device, tokenizer_path=args.tokenizer,
        norm_stats_path=stats_path,
    )
    tables = [pq.read_table(path, columns=["episode_index", "frame_index"])
              for path in sorted((args.dataset_root / "data").rglob("*.parquet"))]
    episodes = np.concatenate([np.asarray(table["episode_index"]) for table in tables])
    frames = np.concatenate([np.asarray(table["frame_index"]) for table in tables])
    assert len(episodes) == len(spec.dataset) and set(episodes) == {0, 1}
    indices = []
    for episode in np.unique(episodes):
        candidates = np.flatnonzero(episodes == episode)
        indices.extend(candidates[np.linspace(0, len(candidates) - 1, 16, dtype=int)].tolist())
    scale = np.asarray(spec.action_stats["q99"]) - np.asarray(spec.action_stats["q01"]) + 1e-6
    errors, hold_errors, first_errors, per_sample = [], [], [], []
    for index in indices:
        sample = spec.dataset[index]
        reference = np.asarray(sample["actions"], dtype=np.float32)
        valid = ~np.asarray(sample["action_is_pad"], dtype=bool)
        request = {key: value for key, value in sample.items()
                   if key not in ("actions", "action_is_pad")}
        noise = np.random.default_rng(1000 + index).standard_normal((50, 32)).astype(np.float32)
        predicted = policy.infer(request, noise=noise)["actions"]
        assert predicted.shape == (50, 6) and np.isfinite(predicted).all() and valid.any()
        error = (predicted - reference)[valid]
        errors.append(error)
        first_errors.append(error[0])
        hold_errors.append((np.asarray(sample["observation/state"])[None] - reference)[valid])
        per_sample.append({"index": index, "episode": int(episodes[index]),
                           "frame": int(frames[index]), "valid_steps": int(valid.sum()),
                           "joint_mae": np.abs(error).mean(axis=0).tolist()})
        np.savez_compressed(args.output / f"frame_{index:06d}.npz",
                            predicted=predicted, reference=reference, valid=valid)
    error = np.concatenate(errors)
    hold = np.concatenate(hold_errors)
    result = {
        "checkpoint": str(args.checkpoint), "samples": len(indices),
        "valid_action_rows": len(error), "fps": 15, "horizon": 50,
        "joint_mae": np.abs(error).mean(axis=0).tolist(),
        "first_action_joint_mae": np.abs(first_errors).mean(axis=0).tolist(),
        "range_normalized_rmse": float(np.sqrt(np.mean((error / scale) ** 2))),
        "hold_state_joint_mae": np.abs(hold).mean(axis=0).tolist(),
        "hold_state_range_normalized_rmse": float(np.sqrt(np.mean((hold / scale) ** 2))),
        "per_sample": per_sample,
        "scope": "32 fixed training observations; not held-out or real-robot evaluation",
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    summary = {key: value for key, value in result.items() if key != "per_sample"}
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

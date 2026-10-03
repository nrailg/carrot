"""Independently check saved predictions against raw Parquet and old paired noise."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--old-eval", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    args = parser.parse_args()
    table = pa.concat_tables([pq.read_table(p, columns=["index", "episode_index", "action"])
                              for p in sorted(args.source_data.rglob("*.parquet"))])
    np.testing.assert_array_equal(np.asarray(table["index"]), np.arange(264))
    np.testing.assert_array_equal(np.asarray(table["episode_index"]), np.zeros(264))
    raw = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    metrics = json.loads((args.evaluation_root / "metrics.json").read_text())
    files = list((args.evaluation_root / "predictions").glob("*.npz"))
    assert len(files) == metrics["samples"] == 292
    expected_keys = {(frame, 1000 + frame + rep * 100000)
                     for frame in range(264)
                     for rep in range(8 if frame in (53, 82, 201, 202) else 1)}
    actual_keys = {(r["frame"], r["noise_seed"]) for r in metrics["per_sample"]}
    assert actual_keys == expected_keys and len(actual_keys) == len(metrics["per_sample"])
    collected = {group: {str(s): [] for s in (1, 5, 10)} for group in ("primary", "probe")}
    old_collected = {str(s): [] for s in (1, 5, 10)}
    noise_hashes, worst = set(), []
    for frame, seed in sorted(actual_keys):
        name = f"frame_{frame:06d}_noise_{seed}.npz"
        reference = raw[np.minimum(np.arange(frame, frame + 10), 263)]
        valid = np.arange(10) < min(10, 264 - frame)
        gaussian = np.random.default_rng(seed).standard_normal((10, 32)).astype(np.float32)
        expected_noise = torch.from_numpy(gaussian).to(torch.bfloat16).float().numpy()
        with np.load(args.evaluation_root / "predictions" / name) as saved:
            np.testing.assert_array_equal(saved["reference"], reference)
            np.testing.assert_array_equal(saved["valid"], valid)
            np.testing.assert_array_equal(saved["noise"], expected_noise)
            assert saved["predicted"].shape == (10, 6) and np.isfinite(saved["predicted"]).all()
            predicted = saved["predicted"].copy()
            noise_hashes.add(hashlib.sha256(saved["noise"].tobytes()).hexdigest())
        with np.load(args.old_eval / name) as old:
            np.testing.assert_array_equal(old["reference"], reference)
            np.testing.assert_array_equal(old["valid"], valid)
            np.testing.assert_array_equal(old["noise"], expected_noise)
            old_prediction = old["predicted"].copy()
        group = "primary" if seed == 1000 + frame else "probe"
        for steps in (1, 5, 10):
            collected[group][str(steps)].append(
                np.abs(predicted[:steps] - reference[:steps])[valid[:steps]])
            if group == "primary":
                old_collected[str(steps)].append(
                    np.abs(old_prediction[:steps] - reference[:steps])[valid[:steps]])
        if group == "primary":
            error = np.abs(predicted - reference)[valid]
            worst.append({"frame": frame, "five_axis_max_degrees": float(error[:, :5].max()),
                          "joint_mae": error.mean(axis=0).tolist()})
    assert len(noise_hashes) == 292
    assert len(collected["primary"]["10"]) == 264
    assert len(collected["probe"]["10"]) == 28
    old_metrics = {}
    for group, chunks in collected.items():
        for steps, values in chunks.items():
            absolute = np.concatenate(values)
            result = metrics[group][steps]
            assert len(absolute) == result["valid_action_rows"]
            np.testing.assert_allclose(absolute.mean(0), result["joint_mae"], rtol=1e-6)
            np.testing.assert_allclose(np.quantile(absolute, .95, axis=0), result["joint_p95"],
                                       rtol=1e-6)
            np.testing.assert_allclose(absolute.max(0), result["joint_max"], rtol=1e-6)
    for steps, values in old_collected.items():
        absolute = np.concatenate(values)
        old_metrics[steps] = {
            "joint_mae": absolute.mean(0).tolist(),
            "joint_p95": np.quantile(absolute, .95, axis=0).tolist(),
            "joint_max": absolute.max(0).tolist(),
        }
    output = {
        "reference_padding_noise": "all 292 exact against raw Parquet and paired old NPZ",
        "unique_noise_tensors": len(noise_hashes), "independent_mae_p95_max": "PASS",
        "old_fp32_primary": old_metrics,
        "new_bf16_primary": metrics["primary"], "new_bf16_probe": metrics["probe"],
        "worst_primary_frames": sorted(worst, key=lambda r: -r["five_axis_max_degrees"])[:10],
    }
    (args.evaluation_root / "independent_validation.json").write_text(
        json.dumps(output, indent=2, allow_nan=False))
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

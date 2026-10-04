"""Recompute denoising diagnostics from raw Parquet and saved latent trajectories."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from recipes.pi05_sft_so101_fit_validation.reevaluate import write_json


def check_stats(values: list[np.ndarray], reported: dict) -> None:
    absolute = np.abs(np.concatenate(values))
    assert len(absolute) == reported["valid_action_rows"]
    for key, computed in (
        ("joint_mae", absolute.mean(0)),
        ("joint_p95", np.quantile(absolute, .95, axis=0)),
        ("joint_max", absolute.max(0)),
    ):
        np.testing.assert_allclose(computed, reported[key], rtol=1e-6, atol=1e-8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--paired-eval", type=Path, required=True)
    args = parser.parse_args()
    metrics = json.loads((args.output / "metrics.json").read_text())
    parquet = sorted(args.source_data.rglob("*.parquet"))
    table = pa.concat_tables([pq.read_table(p, columns=["index", "episode_index", "action"])
                              for p in parquet])
    np.testing.assert_array_equal(np.asarray(table["index"]), np.arange(264))
    np.testing.assert_array_equal(np.asarray(table["episode_index"]), np.zeros(264))
    raw = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    stats = json.loads((Path(metrics["checkpoint"]) / "norm_stats.json").read_text())["action"]
    first, second = [np.asarray(stats[k], dtype=np.float32) for k in ("q01", "q99")]
    span = second - first + np.float32(1e-6)
    scale = ((span + first) - (span / 2 + first))[None]
    expected = {(frame, 1000 + frame + rep * 100000)
                for frame in range(264)
                for rep in range(8 if frame in (53, 82, 201, 202) else 1)}
    keys = {(row["frame"], row["noise_seed"]) for row in metrics["per_sample"]}
    assert keys == expected and len(metrics["per_sample"]) == 292
    files = list((args.output / "predictions").glob("*.npz"))
    assert len(files) == 292
    collected = {group: {str(n): {str(h): [] for h in (1, 5, 10)} for n in metrics["nfe"]}
                 for group in ("primary", "probe")}
    trace_values = {key: [[] for _ in range(10)] for key in metrics["trace_nfe10"]}
    noise_hashes, closure_max, baseline_max = set(), 0.0, 0.0
    signed_sums, unsigned_sums, decomposition = [], [], []
    for frame, seed in sorted(keys):
        name = f"frame_{frame:06d}_noise_{seed}.npz"
        with np.load(args.output / "predictions" / name) as f:
            saved = dict(f)
        reference = raw[np.minimum(np.arange(frame, frame + 10), 263)]
        valid = np.arange(10) < min(10, 264 - frame)
        gaussian = np.random.default_rng(seed).standard_normal((10, 32)).astype(np.float32)
        noise = torch.from_numpy(gaussian).to(torch.bfloat16).float().numpy()
        np.testing.assert_array_equal(saved["reference"], reference)
        np.testing.assert_array_equal(saved["valid"], valid)
        np.testing.assert_array_equal(saved["noise"], noise)
        noise_hashes.add(hashlib.sha256(noise.tobytes()).hexdigest())
        target = np.pad(2 * (reference - first) / span - 1, ((0, 0), (0, 26)))
        np.testing.assert_array_equal(target, saved["normalized_target"])
        with np.load(args.paired_eval / "predictions" / name) as old:
            np.testing.assert_array_equal(old["reference"], reference)
            np.testing.assert_array_equal(old["valid"], valid)
            np.testing.assert_array_equal(old["noise"], noise)
            difference = np.abs(saved["predicted_nfe_10"] - old["predicted"])
            baseline_max = max(baseline_max, float(difference.max()))
            np.testing.assert_allclose(saved["predicted_nfe_10"], old["predicted"],
                                       rtol=0, atol=2e-5)
        latent, velocity, teacher, times = [saved[k] for k in
                                           ("latents", "velocities", "teacher_velocities", "times")]
        assert latent.shape == (11, 10, 32) and velocity.shape == teacher.shape == (10, 10, 32)
        assert times.shape == (10,) and np.isfinite(latent).all()
        np.testing.assert_array_equal(latent[0], noise)
        np.testing.assert_allclose(times, 1 - np.arange(10) / 10, rtol=0, atol=2e-7)
        np.testing.assert_array_equal(latent[1:], latent[:-1] + np.float32(-.1) * velocity)
        closure = float(np.abs(noise - velocity.mean(0, dtype=np.float64) - latent[-1]).max())
        closure_max = max(closure_max, closure)
        assert closure < 2e-6, closure
        decoded = (latent[-1, :, :6] + 1) / 2 * span + first
        np.testing.assert_array_equal(decoded, saved["predicted_nfe_10"])
        group = "primary" if seed == 1000 + frame else "probe"
        for nfe, windows in collected[group].items():
            prediction = saved[f"predicted_nfe_{nfe}"]
            assert prediction.shape == (10, 6) and np.isfinite(prediction).all()
            for h, values in windows.items():
                h = int(h)
                values.append((prediction[:h] - reference[:h])[valid[:h]])
        if group == "primary":
            true_velocity = noise - target
            for step, t in enumerate(times):
                ideal = t * noise + (1 - t) * target
                values = {
                    "drift": latent[step] - ideal,
                    "clean_estimate": latent[step] - t * velocity[step] - target,
                    "teacher_bias": teacher[step] - true_velocity,
                    "rollout_bias": velocity[step] - true_velocity,
                    "feedback": velocity[step] - teacher[step],
                }
                for key, value in values.items():
                    trace_values[key][step].append((value[None, :, :6] * scale)[0, valid])
            increments = -.1 * (velocity[:, :, :5].astype(np.float64)
                                - true_velocity[None, :, :5]) * scale[:, :5]
            signed_sums.append(np.abs(increments.sum(0))[valid])
            unsigned_sums.append(np.abs(increments).sum(0)[valid])
            teacher_term = -.1 * (teacher.astype(np.float64) - true_velocity[None]).sum(0)
            feedback_term = -.1 * (velocity.astype(np.float64) - teacher).sum(0)
            np.testing.assert_allclose(teacher_term + feedback_term,
                                       latent[-1] - target, rtol=0, atol=3e-6)
            decomposition.append({
                "frame": frame, "teacher_term_mae": np.abs(teacher_term[:, :5] * scale[0, :5])
                [valid].mean(0).tolist(),
                "feedback_term_mae": np.abs(feedback_term[:, :5] * scale[0, :5])
                [valid].mean(0).tolist(),
            })
    assert len(noise_hashes) == 292
    for group, nfe_groups in collected.items():
        for nfe, windows in nfe_groups.items():
            for h, values in windows.items():
                check_stats(values, metrics[group][nfe][h])
    for key, steps in trace_values.items():
        for i, values in enumerate(steps):
            check_stats(values, metrics["trace_nfe10"][key][i])
    write_json(args.output / "independent_validation.json", {
        "status": "PASS", "samples": 292, "primary": 264, "probe": 28,
        "unique_noise": len(noise_hashes), "closure_max": closure_max,
        "baseline_max": baseline_max,
        "reference_padding_noise": "exact raw Parquet and seeded noise",
        "mae_p95_max": "all NFE/windows/groups/trace recomputed",
        "signed_error_integral_mae_degrees": np.concatenate(signed_sums).mean(0).tolist(),
        "sum_absolute_increments_degrees": np.concatenate(unsigned_sums).mean(0).tolist(),
        "teacher_feedback_decomposition": decomposition,
        "raw_parquet_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in parquet},
    })
    print("INDEPENDENT_DIFFUSION_VALIDATION_PASS", len(files), "NPZ", flush=True)


if __name__ == "__main__":
    main()

"""Measure Euler rollout drift and paired denoising-step sensitivity."""

import argparse
import importlib.metadata
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

from carrot.data.so101 import build_dataset
from carrot.models.pi05.inference.policy_config import create_so101_policy
from recipes.pi05_sft_so101_fit_validation.fit import fixed_noise
from recipes.pi05_sft_so101_fit_validation.reevaluate import digest, summarize, write_json


@torch.no_grad()
def trace(policy, request: dict, noise: np.ndarray, target: np.ndarray) -> tuple:
    model = policy._model
    original = model.denoise_step
    target_tensor = torch.as_tensor(target, device=policy._device)[None]
    noise_tensor = torch.as_tensor(noise, device=policy._device)[None]
    states, velocities, teachers, times = [], [], [], []

    def observe(state, masks, cache, latent, time):
        velocity = original(state, masks, cache, latent, time)
        # Compare with the training interpolation at the same t, using the same prefix cache.
        ideal = time[:, None, None] * noise_tensor + (1 - time[:, None, None]) * target_tensor
        teacher = original(state, masks, cache, ideal, time)
        states.append(latent[0].float().cpu().numpy().copy())
        velocities.append(velocity[0].float().cpu().numpy().copy())
        teachers.append(teacher[0].float().cpu().numpy().copy())
        times.append(float(time[0]))
        return velocity

    model.denoise_step = observe
    try:
        prediction = policy.infer(request, noise=noise)["actions"]
    finally:
        del model.denoise_step
    assert len(times) == policy.metadata["num_steps"] == 10, times
    final = states[-1] + np.float32(-0.1) * velocities[-1]
    normalized = np.stack(states + [final])
    decoded = policy._output_transform({"actions": normalized})["actions"]
    np.testing.assert_array_equal(decoded[-1], prediction)
    return prediction, normalized, np.stack(velocities), np.stack(teachers), np.asarray(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--paired-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load((args.case_dir / "train.yaml").read_text())
    checkpoint = Path(cfg["output_dir"]) / "checkpoints/step-00005000"
    source = build_dataset(**cfg["dataset"]["factory_kwargs"])
    assert len(source.dataset) == 264, "Expected the complete training episode"
    args.output.mkdir(parents=True, exist_ok=False)
    predictions = args.output / "predictions"
    predictions.mkdir()
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "src/carrot").rglob("*.py"))
    paths += list(Path(__file__).parent.glob("*.py"))
    hashes = {str(path.relative_to(root)): digest(path) for path in paths}
    for path in paths:
        dest = args.output / "source_snapshot" / path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    shutil.copyfile(args.case_dir / "train.yaml", args.output / "train.yaml")
    checkpoint_hashes = {
        name: digest(checkpoint / name)
        for name in ("model.safetensors", "config.json", "norm_stats.json")
    }
    policy = create_so101_policy(
        checkpoint, device="cuda:0", tokenizer_path=cfg["model"]["tokenizer_path"], num_steps=10,
    )
    assert policy._model.config.dtype == "bfloat16"
    assert policy._model.config.action_horizon == 10
    nfe_values = (1, 2, 5, 10, 20, 50)
    groups = {group: {n: {h: [] for h in (1, 5, 10)} for n in nfe_values}
              for group in ("primary", "probe")}
    diagnostics = {key: [[] for _ in range(10)] for key in
                   ("drift", "clean_estimate", "teacher_bias", "rollout_bias", "feedback")}
    closure_max, baseline_max = 0.0, 0.0
    rows = []
    for frame in range(264):
        sample = source.dataset[frame]
        reference = np.asarray(sample["actions"], dtype=np.float32)
        valid = ~np.asarray(sample["action_is_pad"], dtype=bool)
        target = np.asarray(policy._input_transform(dict(sample))["actions"], dtype=np.float32)
        roundtrip = policy._output_transform({"actions": target})["actions"]
        np.testing.assert_allclose(roundtrip, reference, rtol=0, atol=2e-5)
        request = {k: v for k, v in sample.items() if k not in ("actions", "action_is_pad")}
        zero = policy._output_transform({"actions": np.zeros_like(target)})["actions"]
        unit = policy._output_transform({"actions": np.ones_like(target)})["actions"]
        scale = (unit - zero)[None]
        for rep in range(8 if frame in (53, 82, 201, 202) else 1):
            seed = 1000 + frame + rep * 100000
            noise = fixed_noise(seed, 10, 32)
            name = f"frame_{frame:06d}_noise_{seed}.npz"
            with np.load(args.paired_eval / "predictions" / name) as old:
                np.testing.assert_array_equal(noise, old["noise"])
                np.testing.assert_array_equal(reference, old["reference"])
                np.testing.assert_array_equal(valid, old["valid"])
                baseline = old["predicted"].copy()
            label = "primary" if rep == 0 else "probe"
            saved = {"reference": reference, "valid": valid, "noise": noise,
                     "normalized_target": target}
            for nfe in nfe_values:
                policy._num_steps = nfe
                if nfe == 10:
                    prediction, latent, velocity, teacher, times = trace(
                        policy, request, noise, target,
                    )
                    baseline_max = max(baseline_max, float(np.abs(prediction - baseline).max()))
                    np.testing.assert_allclose(prediction, baseline, rtol=0, atol=2e-5)
                    true_velocity = noise - target
                    integrated = noise - np.mean(velocity, axis=0, dtype=np.float64)
                    closure = float(np.abs(integrated - latent[-1]).max())
                    assert closure < 2e-6, closure
                    closure_max = max(closure_max, closure)
                    saved.update(latents=latent, velocities=velocity, teacher_velocities=teacher,
                                 times=times)
                    if label == "primary":
                        for step in range(10):
                            ideal = times[step] * noise + (1 - times[step]) * target
                            clean = latent[step] - times[step] * velocity[step]
                            values = {
                                "drift": (latent[step] - ideal)[None, :, :6] * scale,
                                "clean_estimate": (clean - target)[None, :, :6] * scale,
                                "teacher_bias": (
                                    (teacher[step] - true_velocity)[None, :, :6] * scale
                                ),
                                "rollout_bias": (
                                    (velocity[step] - true_velocity)[None, :, :6] * scale
                                ),
                                "feedback": (velocity[step] - teacher[step])[None, :, :6] * scale,
                            }
                            for key, value in values.items():
                                diagnostics[key][step].append(value[0, valid])
                else:
                    prediction = policy.infer(request, noise=noise)["actions"]
                assert np.isfinite(prediction).all()
                saved[f"predicted_nfe_{nfe}"] = prediction
                for horizon in (1, 5, 10):
                    groups[label][nfe][horizon].append(
                        (prediction[:horizon] - reference[:horizon])[valid[:horizon]],
                    )
            np.savez_compressed(predictions / name, **saved)
            rows.append({"frame": frame, "noise_seed": seed, "group": label})
        if (frame + 1) % 10 == 0 or frame == 263:
            print("PROGRESS", frame + 1, "/264", "samples", len(rows), flush=True)
    summary = {
        "nfe": list(nfe_values), "horizon": 10, "checkpoint": str(checkpoint),
        "samples": len(rows), "per_sample": rows,
        "primary": {str(n): {str(h): summarize(v) for h, v in windows.items()}
                    for n, windows in groups["primary"].items()},
        "probe": {str(n): {str(h): summarize(v) for h, v in windows.items()}
                  for n, windows in groups["probe"].items()},
        "trace_nfe10": {key: [summarize(v) for v in steps]
                        for key, steps in diagnostics.items()},
        "normalized_euler_closure_max": closure_max,
        "baseline_nfe10_max_abs_degrees_or_gripper": baseline_max,
        "units": ["degrees"] * 5 + ["source gripper unit"],
        "diagnostic_scope": "Teacher interpolation is a reference trajectory, not an oracle "
                            "for the learned conditional vector field away from that trajectory.",
    }
    assert len(rows) == 292
    assert summary["primary"]["10"]["10"]["valid_action_rows"] == 2595
    assert checkpoint_hashes == {name: digest(checkpoint / name) for name in checkpoint_hashes}
    write_json(args.output / "metrics.json", summary)
    write_json(args.output / "provenance.json", {
        "source_commit": args.source_commit, "source_hashes": hashes,
        "checkpoint_hashes": checkpoint_hashes, "gpu": torch.cuda.get_device_name(0),
        "docker_image_tag": "未记录", "packages": {
            name: importlib.metadata.version(name) for name in ("torch", "numpy", "transformers")
        }, "protocol": "BF16 production infer/sample_actions; eager prefix/action cache; "
                       "264 unique observations plus 28 separate noise probes; paired noise; "
                       "passive denoise_step observation at NFE10 plus teacher trajectory calls",
    })
    print("DIFFUSION_STEPS_COMPLETE", "samples", len(rows), "baseline_max", baseline_max,
          "closure_max", closure_max, flush=True)


if __name__ == "__main__":
    main()

"""Evaluate a fit case on source observations and explicitly shared latents."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
from transformers import AutoTokenizer

from carrot.data.so101 import build_dataset
from carrot.models.pi05.embodiments.so101 import create_so101_transform_spec
from carrot.models.pi05.inference.policy import Pi05Policy
from carrot.models.pi05.model import PI0Pytorch
from recipes.pi05_sft_so101_fit_validation.fit import DropVision, fixed_noise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, default=10)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    fit = cfg["fit"]
    kwargs = dict(cfg["dataset"]["factory_kwargs"])
    selected = kwargs.pop("frame_index")
    kwargs.pop("repeat_count")
    source = build_dataset(**kwargs)
    horizon = kwargs["action_horizon"]
    stats = {"state": source.state_stats, "actions": source.action_stats}
    model = PI0Pytorch.from_pretrained(args.checkpoint)
    if args.checkpoint == Path(cfg["model"]["path"]):
        model.register_to_config(action_horizon=horizon)
    assert model.config.action_horizon == horizon
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["tokenizer_path"],
                                             local_files_only=True, fix_mistral_regex=True)
    spec = create_so101_transform_spec(tokenizer, stats, model_action_dim=model.config.action_dim,
                                       discrete_state_input=model.config.discrete_state_input)
    if not fit["vision"]:
        spec = replace(spec, inputs=spec.inputs + (DropVision(),))
    policy = Pi05Policy(model, spec, device="cuda:0", num_steps=args.num_steps)
    indices = list(range(len(source.dataset))) if selected is None else [selected]
    probes = {53, 82, 201, 202} if selected is None else {selected}
    args.output.mkdir(parents=True, exist_ok=False)
    errors, short_errors, rows = [], [], []
    for index in indices:
        sample = source.dataset[index]
        reference = np.asarray(sample["actions"])
        valid = ~np.asarray(sample["action_is_pad"], dtype=bool)
        request = {k: v for k, v in sample.items() if k not in ("actions", "action_is_pad")}
        for repetition in range(8 if index in probes else 1):
            seed = 1000 + index + repetition * 100000
            noise = fixed_noise(seed, horizon, model.config.action_dim)
            prediction = policy.infer(request, noise=noise)["actions"]
            assert prediction.shape == reference.shape == (horizon, 6)
            assert np.isfinite(prediction).all()
            error = (prediction - reference)[valid]
            errors.append(error)
            short_errors.append((prediction[:5] - reference[:5])[valid[:5]])
            rows.append({"frame": index, "noise_seed": seed, "valid_steps": int(valid.sum()),
                         "joint_mae": np.abs(error).mean(axis=0).tolist()})
            np.savez_compressed(args.output / f"frame_{index:06d}_noise_{seed}.npz",
                                predicted=prediction, reference=reference, valid=valid, noise=noise)
    error = np.concatenate(errors)
    short = np.concatenate(short_errors)
    scale = np.asarray(source.action_stats["q99"]) - np.asarray(source.action_stats["q01"]) + 1e-6
    metrics = {"unique_frames": len(indices), "samples": len(rows), "horizon": horizon,
               "vision": fit["vision"], "fixed_training_noise_seed": fit["noise_seed"],
               "denoising_steps": args.num_steps, "valid_action_rows": len(error),
               "joint_mae": np.abs(error).mean(axis=0).tolist(),
               "joint_p95": np.quantile(np.abs(error), 0.95, axis=0).tolist(),
               "joint_max": np.abs(error).max(axis=0).tolist(),
               "first_five_joint_mae": np.abs(short).mean(axis=0).tolist(),
               "range_normalized_rmse": float(np.sqrt(np.mean((error / scale) ** 2))),
               "per_sample": rows}
    if fit["noise_seed"] is not None:
        fixed = next(row for row in rows if row["noise_seed"] == fit["noise_seed"])
        metrics["training_noise_joint_mae"] = fixed["joint_mae"]
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    print(
        "EVALUATION", json.dumps({k: v for k, v in metrics.items() if k != "per_sample"}),
        flush=True,
    )


if __name__ == "__main__":
    main()

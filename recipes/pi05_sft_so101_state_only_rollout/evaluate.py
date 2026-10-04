"""Measure state-feedback drift across consecutive SO101 diffusion processes."""

import argparse
import importlib.metadata
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from carrot.data.so101 import build_dataset
from carrot.models.pi05 import transforms
from carrot.models.pi05.inference.policy_config import create_so101_policy
from recipes.pi05_sft_so101_fit_validation.fit import DropVision, fixed_noise
from recipes.pi05_sft_so101_fit_validation.reevaluate import digest, summarize, write_json

MODES = ("recorded_state", "predicted_action_state")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load((args.case_dir / "train.yaml").read_text())
    assert cfg["steps"] == 500 and cfg["fit"] == {"vision": False, "noise_seed": None}
    kwargs = dict(cfg["dataset"]["factory_kwargs"])
    jitter = kwargs.pop("state_jitter_degrees", 0.0)
    assert (cfg["dataset"]["factory"], jitter) in (
        ("carrot.data.so101.build_dataset", 0.0),
        ("recipes.pi05_sft_so101_state_jitter.augmentation.build_dataset", 3.0),
    ), "evaluate clean source states; training jitter is a separate condition"
    source = build_dataset(**kwargs)
    assert len(source.dataset) == 264
    checkpoint = Path(cfg["output_dir"]) / "checkpoints/step-00000500"
    checkpoint_hashes = {name: digest(checkpoint / name) for name in
                         ("model.safetensors", "config.json", "norm_stats.json")}
    args.output.mkdir(parents=True, exist_ok=False)
    predictions = args.output / "predictions"
    predictions.mkdir()
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "src/carrot").rglob("*.py"))
    paths += list(Path(__file__).parent.glob("*.py"))
    paths += list((root / "recipes/pi05_sft_so101_fit_validation").glob("*.py"))
    paths += list(args.case_dir.resolve().glob("*.py"))
    hashes = {str(path.relative_to(root)): digest(path) for path in paths}
    for path in paths:
        target = args.output / "source_snapshot" / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    shutil.copyfile(args.case_dir / "train.yaml", args.output / "train.yaml")
    policy = create_so101_policy(
        checkpoint, device="cuda:0", tokenizer_path=cfg["model"]["tokenizer_path"], num_steps=10,
    )
    assert policy._model.config.dtype == "bfloat16" and policy.metadata["action_horizon"] == 10
    spec = replace(policy._transform_spec, inputs=policy._transform_spec.inputs + (DropVision(),))
    policy._transform_spec = spec
    policy._input_transform = transforms.compose(spec.inputs)
    write_json(args.output / "loaded_dtypes.json", {
        name: str(value.dtype) for name, value in policy._model.named_parameters()
    })
    groups = {mode: {depth: [] for depth in range(53)} for mode in MODES}
    endpoint_errors = {mode: {depth: [] for depth in range(53)} for mode in MODES}
    state_errors = {mode: {depth: [] for depth in range(53)} for mode in MODES}
    rows = []
    for repetition in range(8):
        previous_predictions = {}
        for depth, frame in enumerate(range(0, 264, 5)):
            sample = source.dataset[frame]
            recorded = np.asarray(sample["observation/state"], dtype=np.float32)
            reference = np.asarray(sample["actions"], dtype=np.float32)
            valid = ~np.asarray(sample["action_is_pad"], dtype=bool)
            executed = valid.copy()
            executed[5:] = False
            count = int(executed.sum())
            assert count == min(5, 264 - frame)
            seed = 1000 + frame + repetition * 100000
            noise = fixed_noise(seed, 10, 32)
            states = {mode: recorded.copy() for mode in MODES}
            if depth:
                states["predicted_action_state"] = previous_predictions[
                    "predicted_action_state"
                ].copy()
            saved = {"recorded_state": recorded, "reference": reference, "valid": valid,
                     "noise": noise, "executed_valid": executed}
            chunk_predictions = {}
            for mode in MODES:
                request = {"observation/state": states[mode].copy(),
                           "prompt": sample["prompt"],
                           "observation/wrist_image": np.zeros((3, 224, 224), dtype=np.uint8)}
                masked = policy._input_transform(dict(request))
                assert not any(masked["image_mask"].values())
                assert all(not np.asarray(image).any() for image in masked["image"].values())
                prediction = policy.infer(request, noise=noise)["actions"]
                assert prediction.shape == (10, 6) and np.isfinite(prediction).all()
                chunk_predictions[mode] = prediction
                saved[f"input_{mode}"] = states[mode]
                saved[f"predicted_{mode}"] = prediction
                groups[mode][depth].append((prediction - reference)[executed])
                endpoint_errors[mode][depth].append(prediction[count - 1] - reference[count - 1])
                state_errors[mode][depth].append(states[mode] - recorded)
            if not depth:
                for mode in MODES:
                    np.testing.assert_array_equal(chunk_predictions[mode],
                                                  chunk_predictions["recorded_state"])
            np.savez_compressed(predictions / f"chain_{repetition:02d}_chunk_{depth:02d}.npz",
                                **saved)
            rows.append({"chain": repetition, "chunk": depth, "frame": frame, "seed": seed,
                         "valid_actions": count})
            previous_predictions = {mode: values[count - 1].copy()
                                    for mode, values in chunk_predictions.items()}
        print("CHAIN_COMPLETE", repetition + 1, "/8", "chunks", depth + 1, flush=True)
    metrics = {
        "checkpoint": str(checkpoint), "chains": 8, "chunks_per_chain": 53,
        "samples": len(rows), "nfe": 10, "horizon": 10, "execute_steps": 5,
        "per_sample": rows,
        "modes": {mode: {
            "chunk_errors": {str(k): summarize(v) for k, v in groups[mode].items()},
            "endpoint_errors": {str(k): summarize([np.stack(v)])
                                for k, v in endpoint_errors[mode].items()},
            "input_state_errors": {str(k): summarize([np.stack(v)])
                                   for k, v in state_errors[mode].items()},
            "whole_episode": summarize([np.concatenate(v) for v in groups[mode].values()]),
        } for mode in MODES},
        "vision": False, "training_state_jitter_degrees": jitter,
        "units": ["degrees"] * 5 + ["source gripper unit"],
        "scope": "8 noise sequences over one recorded episode. State-only offline feedback; "
                 "all image masks false, pixels zero; predicted executed endpoint as next state; "
                 "ideal position tracking, no robot dynamics, no real robot.",
    }
    assert len(rows) == 424
    assert checkpoint_hashes == {name: digest(checkpoint / name) for name in checkpoint_hashes}
    write_json(args.output / "metrics.json", metrics)
    write_json(args.output / "provenance.json", {
        "source_commit": args.source_commit, "source_hashes": hashes,
        "checkpoint_hashes": checkpoint_hashes, "gpu": torch.cuda.get_device_name(0),
        "docker_image_tag": "未记录", "packages": {
            name: importlib.metadata.version(name) for name in ("torch", "numpy", "transformers")
        }, "noise": "1000+frame+chain*100000, fresh per chunk, paired across two modes",
        "transition": "next state = preceding executed prediction at index4; no state reset; "
                      "horizon10/execute5/NFE10, all vision masked in train and inference",
    })
    print("CHUNK_ROLLOUT_COMPLETE", "NPZ", len(rows), "vision", False, flush=True)


if __name__ == "__main__":
    main()

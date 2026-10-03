"""Reevaluate the vision-enabled episode through the production BF16 policy."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors import safe_open

from carrot.data.so101 import build_dataset
from carrot.models.pi05.inference.policy_config import create_so101_policy
from carrot.models.pi05.model import PI0Pytorch
from recipes.pi05_sft_so101_fit_validation.fit import fixed_noise


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False))


def summarize(errors: list[np.ndarray]) -> dict:
    absolute = np.abs(np.concatenate(errors))
    return {
        "valid_action_rows": len(absolute),
        "joint_mae": absolute.mean(axis=0).tolist(),
        "joint_p95": np.quantile(absolute, 0.95, axis=0).tolist(),
        "joint_max": absolute.max(axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load((args.case_dir / "train.yaml").read_text())
    kwargs = dict(cfg["dataset"]["factory_kwargs"])
    assert kwargs.pop("frame_index") is None
    kwargs.pop("repeat_count")
    assert cfg["fit"] == {"vision": True, "noise_seed": None}
    assert kwargs["action_horizon"] == 10 and cfg["steps"] == 1000
    checkpoint = Path(cfg["output_dir"]) / "checkpoints/step-00001000"
    assert json.loads((checkpoint / "trainer_state.json").read_text())["step"] == 1000
    args.output.mkdir(parents=True, exist_ok=False)
    export = args.output / "bf16_checkpoint"
    predictions = args.output / "predictions"
    predictions.mkdir()
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    root = Path(__file__).resolve().parents[2]
    source_paths = list((root / "src/carrot").rglob("*.py"))
    source_paths += list(Path(__file__).parent.glob("*.py"))
    source_paths += [Path(__file__).parent / "reevaluate.sh"]
    hashes = {str(p.relative_to(root)): digest(p) for p in source_paths}
    for path in source_paths:
        target = snapshot / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    shutil.copyfile(args.case_dir / "train.yaml", args.output / "train.yaml")
    original_hashes = {
        name: digest(checkpoint / name)
        for name in ("model.safetensors", "config.json", "norm_stats.json")
    }
    write_json(args.output / "provenance.json", {
        "source_commit": args.source_commit, "source_hashes": hashes,
        "original_checkpoint": str(checkpoint), "original_hashes": original_hashes,
        "docker_image_tag": "未记录", "gpu": torch.cuda.get_device_name(0),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "lerobot", "numpy")},
        "protocol": "264 unique frames; 28 additional noise probes separate; NFE10; eager cache",
        "noise": "same distinct per-frame seeded BF16-rounded Gaussian as old FP32 evaluation",
    })
    print("EXPORT_START", str(checkpoint), flush=True)
    model = PI0Pytorch.from_pretrained(checkpoint)
    model.save_pretrained(export)
    del model
    gc.collect()
    shutil.copyfile(checkpoint / "norm_stats.json", export / "norm_stats.json")
    assert json.loads((export / "config.json").read_text())["precision"] == "bfloat16"
    with safe_open(export / "model.safetensors", framework="pt", device="cpu") as stream:
        stored_dtypes = {stream.get_slice(key).get_dtype() for key in stream.keys()}  # noqa: SIM118
    assert stored_dtypes == {"BF16"}, stored_dtypes
    source = build_dataset(**kwargs)
    assert len(source.dataset) == 264
    stats = json.loads((export / "norm_stats.json").read_text())
    for target, source_stats in (("state", source.state_stats), ("action", source.action_stats)):
        for name in source_stats:
            np.testing.assert_array_equal(stats[target][name], source_stats[name])
    policy = create_so101_policy(
        export, device="cuda:0", tokenizer_path=cfg["model"]["tokenizer_path"], num_steps=10,
    )
    assert policy._model.config.action_horizon == 10
    loaded_dtypes = {name: str(value.dtype) for name, value in policy._model.named_parameters()}
    write_json(args.output / "loaded_dtypes.json", loaded_dtypes)
    groups = {label: {steps: [] for steps in (1, 5, 10)} for label in ("primary", "probe")}
    rows = []
    for index in range(264):
        sample = source.dataset[index]
        reference = np.asarray(sample["actions"])
        valid = ~np.asarray(sample["action_is_pad"], dtype=bool)
        request = {k: v for k, v in sample.items() if k not in ("actions", "action_is_pad")}
        transformed = policy._input_transform(dict(request))
        assert transformed["image_mask"] == {
            "base_0_rgb": False, "left_wrist_0_rgb": True, "right_wrist_0_rgb": False,
        }
        assert np.asarray(transformed["image"]["left_wrist_0_rgb"]).std() > 0
        for repetition in range(8 if index in (53, 82, 201, 202) else 1):
            seed = 1000 + index + repetition * 100000
            noise = fixed_noise(seed, 10, policy._model.config.action_dim)
            name = f"frame_{index:06d}_noise_{seed}.npz"
            # 噪声仅与旧评估配对，每个frame及probe均不同，不采用固定训练noise。
            with np.load(args.case_dir / "evaluation/step1000" / name) as old:
                np.testing.assert_array_equal(noise, old["noise"])
                np.testing.assert_array_equal(reference, old["reference"])
                np.testing.assert_array_equal(valid, old["valid"])
            prediction = policy.infer(request, noise=noise)["actions"]
            assert prediction.shape == reference.shape == (10, 6)
            assert np.isfinite(prediction).all()
            label = "primary" if repetition == 0 else "probe"
            for steps in (1, 5, 10):
                groups[label][steps].append((prediction[:steps] - reference[:steps])[valid[:steps]])
            rows.append({"frame": index, "noise_seed": seed, "group": label})
            np.savez_compressed(predictions / name, predicted=prediction, reference=reference,
                                valid=valid, noise=noise)
        if (index + 1) % 20 == 0 or index == 263:
            print("PROGRESS", index + 1, "/264", "predictions", len(rows), flush=True)
    metrics = {
        "unique_frames": 264, "samples": len(rows), "vision": True,
        "training_noise_seed": None, "horizon": 10, "denoising_steps": 10,
        "units": ["degrees"] * 5 + ["source gripper unit"],
        "primary": {str(s): summarize(groups["primary"][s]) for s in (1, 5, 10)},
        "probe": {str(s): summarize(groups["probe"][s]) for s in (1, 5, 10)},
        "per_sample": rows,
    }
    assert len(rows) == 292 and metrics["primary"]["10"]["valid_action_rows"] == 2595
    assert metrics["probe"]["10"]["valid_action_rows"] == 280
    expert = policy._model.paligemma_with_expert
    assert expert.paligemma.language_model.config._attn_implementation == "eager"
    assert expert.gemma_expert.model.config._attn_implementation == "eager"
    assert original_hashes == {name: digest(checkpoint / name) for name in original_hashes}
    write_json(args.output / "metrics.json", metrics)
    print("EVALUATION_COMPLETE",
          json.dumps({k: v for k, v in metrics.items() if k != "per_sample"}), flush=True)


if __name__ == "__main__":
    main()

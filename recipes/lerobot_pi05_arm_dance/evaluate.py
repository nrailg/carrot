import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from common import VENV, load_dataset
from configuration_pi05_control import PI05ControlConfig
from state_jitter import StateJitterStep
from lerobot.configs import PreTrainedConfig
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.random_utils import set_seed


def metrics(error: np.ndarray) -> dict:
    error = np.abs(error)
    return {
        "mae": error.mean(axis=0).tolist(),
        "p50": np.quantile(error, 0.5, axis=0).tolist(),
        "p90": np.quantile(error, 0.9, axis=0).tolist(),
        "max": error.max(axis=0).tolist(),
    }


def verify_weights(policy, checkpoint: Path) -> int:
    parameters = policy.state_dict()
    count = 0
    with safe_open(checkpoint / "model.safetensors", framework="pt") as archive:
        mapped_keys = {key if key.startswith("model.") else f"model.{key}" for key in archive.keys()}
        missing = set(parameters) - mapped_keys
        head = "model.paligemma_with_expert.paligemma.lm_head.weight"
        embedding = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        assert missing in [set(), {head}, {embedding}], missing
        assert mapped_keys <= set(parameters), mapped_keys - set(parameters)
        if missing:
            # safetensors omits the tied embedding alias; the official loader restores it.
            assert parameters[head].data_ptr() == parameters[embedding].data_ptr()
        assert len(mapped_keys) + len(missing) == len(parameters)
        for key in archive.keys():
            model_key = key if key.startswith("model.") else f"model.{key}"
            assert model_key in parameters, model_key
            actual = parameters[model_key].detach().cpu()
            expected = archive.get_tensor(key).to(actual.dtype)
            assert torch.equal(actual, expected), f"Weight load differs: {key}"
            assert torch.isfinite(actual).all(), model_key
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=144)
    args = parser.parse_args()
    assert Path(sys.prefix) == VENV
    assert args.base != (args.checkpoint is not None), "Choose --base or --checkpoint"
    assert StateJitterStep is not None
    set_seed(1000)
    checkpoint = Path(os.environ["INITIAL_MODEL_DIR"]) if args.base else args.checkpoint
    assert checkpoint.is_dir(), checkpoint
    args.output.mkdir(parents=True, exist_ok=False)
    if args.base:
        config = PI05ControlConfig(
            empty_cameras=2,
            pretrained_path=checkpoint,
            chunk_size=10,
            n_action_steps=1,
            dtype="bfloat16",
            device="cuda",
            push_to_hub=False,
        )
    else:
        config = PreTrainedConfig.from_pretrained(checkpoint)
        config.pretrained_path = checkpoint
        config.device = "cuda"
    assert config.chunk_size == 10 and config.n_action_steps == 1
    dataset = load_dataset(config)
    policy = make_policy(config, ds_meta=dataset.meta)
    verified = verify_weights(policy, checkpoint)
    print(f"WEIGHTS VERIFIED: {verified}", flush=True)
    processor_kwargs = {"preprocessor_overrides": {
        "device_processor": {"device": "cuda"},
        "arm_dance_state_jitter": {"enabled": False},
    }}
    if args.base:
        processor_kwargs["dataset_stats"] = dataset.meta.stats
        processor_kwargs["preprocessor_overrides"].update({
            "normalizer_processor": {
                "features": {**config.input_features, **config.output_features},
                "norm_map": config.normalization_mapping,
                "stats": dataset.meta.stats,
            },
        })
        processor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "features": config.output_features,
                "norm_map": config.normalization_mapping,
                "stats": dataset.meta.stats,
            }
        }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(checkpoint), **processor_kwargs
    )
    predictions, targets, padding, states = [], [], [], []
    policy.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(min(args.limit, len(dataset))):
            sample = dataset[i]
            model_sample = sample.copy()
            model_sample["observation.images.wrist"] = (
                model_sample["observation.images.wrist"].float() / 255
            )
            batch = preprocessor(model_sample)
            images, masks = policy._preprocess_images(batch)
            assert len(images) == 3 and all(torch.count_nonzero(image) == 0 for image in images)
            assert all(not mask.any() for mask in masks)
            policy.reset()
            prediction = postprocessor(policy.predict_action_chunk(batch)).cpu().float().numpy()[0]
            assert prediction.shape == (10, 6) and np.isfinite(prediction).all()
            predictions.append(prediction)
            targets.append(sample["action"].cpu().numpy())
            padding.append(sample["action_is_pad"].cpu().numpy())
            states.append(sample["observation.state"].cpu().numpy())
            print(f"PLAY {i + 1}/{min(args.limit, len(dataset))}", flush=True)
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    valid = ~np.asarray(padding, dtype=bool)
    assert valid[:, 0].all()
    error = predictions - targets
    result = {
        "status": "passed",
        "protocol": "teacher_forced_recorded_state_no_vision_no_jitter",
        "checkpoint": str(checkpoint),
        "frames": len(predictions),
        "valid_chunk_actions": int(valid.sum()),
        "weights_verified": verified,
        "axes": dataset.meta.features["action"]["names"],
        "units": ["degrees"] * 5 + ["gripper_recorded_units"],
        "first_action": metrics(error[:, 0]),
        "valid_chunk": metrics(error[valid]),
        "five_axis_first_action_mae_degrees": float(np.abs(error[:, 0, :5]).mean()),
        "seed": 1000,
        "robot_operated": False,
    }
    assert not any(name == "carrot" or name.startswith("carrot.") for name in sys.modules)
    np.savez_compressed(
        args.output / "predictions.npz", predictions=predictions, targets=targets,
        valid=valid, states=np.asarray(states),
    )
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

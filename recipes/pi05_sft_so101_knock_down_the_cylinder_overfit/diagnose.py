"""Audit recorded SO101 inputs and evaluate the served policy without robot access."""

import argparse
import json
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from transformers import AutoTokenizer

from carrot.data.so101 import build_dataset
from carrot.data.so101_units import model_stats, to_model_units
from carrot.models.pi05 import transforms
from carrot.models.pi05.embodiments.so101 import create_so101_transform_spec
from carrot.models.pi05.inference.policy_config import _load_norm_stats
from examples.so101_real.client import PolicyClient
from examples.so101_real.config import JOINT_NAMES, DeploymentConfig
from examples.so101_real.dataset import load_dataset_source


def _window_metrics(predicted: np.ndarray, reference: np.ndarray,
                    valid: np.ndarray, count: int) -> dict[str, Any]:
    error = (predicted[:, :count] - reference[:, :count])[valid[:, :count]]
    assert len(error) > 0 and np.isfinite(error).all()
    return {
        "rows": len(error),
        "joint_mae": np.abs(error).mean(axis=0).tolist(),
        "joint_p95_abs_error": np.quantile(np.abs(error), 0.95, axis=0).tolist(),
        "joint_bias": error.mean(axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--previous-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-joint-units", choices=("degrees", "radians"), required=True)
    parser.add_argument("--server-uri", default="ws://127.0.0.1:8080")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "arguments.json").write_text(json.dumps(
        {key: str(value) for key, value in vars(args).items()}, indent=2,
    ))
    config = json.loads((args.checkpoint / "config.json").read_text())
    horizon = config["action_horizon"]
    assert horizon == 50 and config["action_dim"] == 32
    dataset_config = yaml.safe_load(Path(__file__).with_name("train.yaml").read_text())[
        "dataset"
    ]["factory_kwargs"]
    spec = build_dataset(**(dataset_config | {"root": str(args.dataset_root)}))
    metadata = spec.dataset.source.meta
    assert metadata.fps == 15 and metadata.total_episodes == 2 and len(spec.dataset) == 354
    for key in ("observation.state", "action"):
        assert tuple(metadata.features[key]["names"]) == JOINT_NAMES, key
    dataset_stats = {"state": spec.state_stats, "actions": spec.action_stats}
    checkpoint_stats = _load_norm_stats(args.checkpoint / "norm_stats.json")
    if args.checkpoint_joint_units == "degrees":
        checkpoint_stats = {key: model_stats(stats, use_degrees=True)
                            for key, stats in checkpoint_stats.items()}
    differences = {}
    for key, stats in dataset_stats.items():
        assert set(stats) == set(checkpoint_stats[key]), key
        for name, values in stats.items():
            a, b = np.asarray(values), np.asarray(checkpoint_stats[key][name])
            assert a.shape == b.shape, (key, name)
            differences[f"{key}.{name}"] = float(np.max(np.abs(a - b)))
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, fix_mistral_regex=True, local_files_only=True,
    )
    # OpenPI checkpoint loader 固定 pi05=True，离散 state 输入随之默认开启。
    train_transform = transforms.compose(create_so101_transform_spec(
        tokenizer, dataset_stats, model_action_dim=32,
        discrete_state_input=True,
    ).inputs)
    serving_transform = transforms.compose(create_so101_transform_spec(
        tokenizer, checkpoint_stats, model_action_dim=32,
        discrete_state_input=True,
    ).inputs)
    table = pa.concat_tables([
        pq.read_table(path, columns=["episode_index", "frame_index", "timestamp",
                                     "observation.state", "action"])
        for path in sorted((args.dataset_root / "data").rglob("*.parquet"))
    ])
    episodes = np.asarray(table["episode_index"])
    frames = np.asarray(table["frame_index"])
    timestamps = np.asarray(table["timestamp"])
    states = np.stack(table["observation.state"].to_pylist()).astype(np.float32)
    raw_actions = np.stack(table["action"].to_pylist()).astype(np.float32)
    states = to_model_units(states, use_degrees=True)
    raw_actions = to_model_units(raw_actions, use_degrees=True)
    assert len(episodes) == 354
    timestamp_error = float(np.max(np.abs(timestamps - frames / metadata.fps)))
    assert timestamp_error < 1e-6, timestamp_error
    requests, references, masks = [], [], []
    for episode in sorted(set(episodes.tolist())):
        indices = np.flatnonzero(episodes == episode)
        assert np.array_equal(frames[indices], np.arange(len(indices)))
        source = load_dataset_source(DeploymentConfig(
            dataset_root=str(args.dataset_root),
            dataset_repo="nrailg/so101_knock_down_the_cylinder",
            episode=int(episode), fps=15, use_degrees=True, base_camera=None,
            wrist_camera="wrist", prompt="Move an object",
        ), horizon)
        for index in indices:
            sample = spec.dataset[int(index)]
            frame = source.read()
            assert frame is not None and frame.episode == episode and frame.frame == frames[index]
            valid = np.arange(horizon) < frame.valid_steps
            expected = raw_actions[index:index + frame.valid_steps]
            assert np.array_equal(np.asarray(sample["action_is_pad"]), ~valid)
            assert np.array_equal(np.asarray(sample["actions"])[valid], expected)
            assert np.array_equal(frame.reference, expected)
            assert np.array_equal(frame.request["observation/state"], states[index])
            assert frame.request["prompt"] == sample["prompt"]
            a, b = train_transform(sample), serving_transform(frame.request)
            for key in ("state", "tokenized_prompt", "tokenized_prompt_mask"):
                assert np.array_equal(a[key], b[key]), (int(index), key)
            assert a["image_mask"] == b["image_mask"]
            for key in a["image"]:
                assert np.array_equal(a["image"][key], b["image"][key]), (int(index), key)
            requests.append(frame.request)
            references.append(np.asarray(sample["actions"], dtype=np.float32))
            masks.append(valid)
            source.advance(1)
            if len(requests) % 50 == 0:
                print(f"audit_frames={len(requests)}/354", flush=True)
        assert source.read() is None
    audit = {
        "frames": len(requests), "fps": metadata.fps, "joint_order": JOINT_NAMES,
        "units": ["radians"] * 5 + ["fraction"],
        "stats_max_abs_differences": differences,
        "timestamp_max_abs_error_s": timestamp_error,
        "reference_action_alignment": "exact for all valid actions",
        "processed_training_vs_deployment_inputs": "exact for all 354 frames",
        "dataset_info_sha256": sha256(
            (args.dataset_root / "meta/info.json").read_bytes()).hexdigest(),
        "checkpoint_stats_sha256": sha256(
            (args.checkpoint / "norm_stats.json").read_bytes()).hexdigest(),
        "checkpoint_config_sha256": sha256(
            (args.checkpoint / "config.json").read_bytes()).hexdigest(),
        "discrete_state_input": True,
    }
    (args.output / "input_audit.json").write_text(json.dumps(audit, indent=2))
    assert max(differences.values()) == 0, "checkpoint and dataset stats differ"
    np.savez_compressed(args.output / "recorded_inputs.npz", states=states,
                        episodes=episodes, frames=frames, reference=np.stack(references),
                        valid=np.stack(masks))
    (args.output / "checkpoint_norm_stats.json").write_text(json.dumps(checkpoint_stats, indent=2))
    previous = sorted(args.previous_evaluation.glob("frame_*.npz"))
    assert len(previous) == 32
    old = [np.load(path) for path in previous]
    previous_metrics = {str(count): _window_metrics(
        to_model_units(np.stack([x["predicted"] for x in old]), use_degrees=True),
        to_model_units(np.stack([x["reference"] for x in old]), use_degrees=True),
        np.stack([x["valid"] for x in old]), count,
    ) for count in (1, 5, 50)}
    (args.output / "previous_32_window_metrics.json").write_text(
        json.dumps(previous_metrics, indent=2))
    for data in old:
        data.close()

    client = PolicyClient(args.server_uri, 10, 15)
    predicted, inference_ms = [], []
    try:
        assert client.metadata["embodiment"] == "so101"
        assert client.metadata["joint_units"] == "radians"
        assert client.metadata["gripper_units"] == "fraction"
        assert client.metadata["action_horizon"] == horizon and client.metadata["action_dim"] == 6
        assert client.metadata["num_steps"] == 10
        (args.output / "server_metadata.json").write_text(json.dumps(client.metadata, indent=2))
        for index, request in enumerate(requests):
            start = time.monotonic()
            response = client.infer(request)
            elapsed_ms = (time.monotonic() - start) * 1000
            actions = np.asarray(response["actions"])
            assert actions.dtype == np.float32 and actions.shape == (horizon, 6)
            assert np.isfinite(actions).all()
            predicted.append(actions)
            inference_ms.append(elapsed_ms)
            np.savez_compressed(args.output / f"frame_{index:06d}.npz", predicted=actions,
                                reference=references[index], valid=masks[index],
                                state=states[index])
            if (index + 1) % 25 == 0:
                print(f"inferred_frames={index + 1}/354", flush=True)
    finally:
        client.close()
    predicted, reference, valid = np.stack(predicted), np.stack(references), np.stack(masks)
    metrics = {str(count): _window_metrics(predicted, reference, valid, count)
               for count in (1, 5, 50)}
    by_episode = {
        str(episode): {str(count): _window_metrics(
            predicted[episodes == episode], reference[episodes == episode],
            valid[episodes == episode], count,
        ) for count in (1, 5, 50)} for episode in sorted(set(episodes.tolist()))
    }
    summary = {
        "frames": len(predicted), "episodes": len(by_episode), "windows": metrics,
        "by_episode": by_episode, "median_request_ms": float(np.median(inference_ms)),
        "scope": "teacher-forced training observations; one random-noise draw per frame; no robot",
        "server_uri": args.server_uri,
    }
    np.savez_compressed(args.output / "all_predictions.npz", predicted=predicted,
                        reference=reference, valid=valid, states=states, frames=frames,
                        episodes=episodes, request_ms=np.asarray(inference_ms))
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

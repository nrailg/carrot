"""Capture one real LIBERO sample and its official OpenPI PyTorch policy outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from openpi.models import model as openpi_model
from openpi.policies.policy_config import create_trained_policy
from openpi.shared import normalize
from openpi.training.config import get_config

IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
OPENPI_COMMIT = "89d9220c68defd85b85f17acf130e965e58b8369"
OPENPI_SOURCE_SHA256 = "3b5e87f546e2e8effe3dac5b54c42f103bac7dc159c30047f4745ddaf5e5264d"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numpy(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _source_sha256(openpi_dir: Path) -> str:
    paths = sorted(
        [*openpi_dir.joinpath("src/openpi").rglob("*.py"), openpi_dir / "examples/libero/main.py"],
        key=lambda path: str(path.relative_to(openpi_dir)),
    )
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("OpenPI source tree is incomplete")
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(openpi_dir)
        digest.update(f"{_sha256(path)}  {relative}\n".encode())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpi-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="lerobot/libero")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_sha256 = _source_sha256(args.openpi_dir)
    if source_sha256 != OPENPI_SOURCE_SHA256:
        raise ValueError(f"OpenPI source fingerprint mismatch: {source_sha256}")
    if args.num_steps < 1:
        raise ValueError("num_steps must be positive")
    stats_path = args.checkpoint / "assets/physical-intelligence/libero/norm_stats.json"
    weight_path = args.checkpoint / "model.safetensors"
    for path in (stats_path, weight_path, args.checkpoint / "config.json"):
        if not path.is_file():
            raise FileNotFoundError(path)

    dataset_meta = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(args.sample_index)
    sample = dataset[args.sample_index]
    prompt = dataset_meta.tasks[int(sample["task_index"])]
    raw = {
        "observation/state": _numpy(sample["observation/state"]).astype(np.float32),
        "observation/image": _numpy(sample["observation/image"]),
        "observation/wrist_image": _numpy(sample["observation/wrist_image"]),
        "prompt": prompt,
    }
    if raw["observation/state"].shape != (8,):
        raise ValueError("LIBERO state must have shape (8,)")

    stats = normalize.load(stats_path.parent)
    config = get_config("pi05_libero")
    assert config.model.action_horizon == 10 and config.model.action_dim == 32
    policy = create_trained_policy(
        config,
        args.checkpoint,
        norm_stats=stats,
        sample_kwargs={"num_steps": args.num_steps},
        pytorch_device="cuda",
    )
    transformed = policy._input_transform(dict(raw))
    inputs = jax.tree.map(
        lambda value: torch.from_numpy(np.asarray(value).copy()).to("cuda")[None, ...],
        transformed,
    )
    observation = openpi_model.Observation.from_dict(inputs)
    noise = np.random.default_rng(7).standard_normal((10, 32)).astype(np.float32)
    with torch.no_grad():
        actions = policy._model.sample_actions(
            "cuda",
            observation,
            noise=torch.from_numpy(noise[None]).to("cuda"),
            num_steps=args.num_steps,
        )
    if actions.shape != (1, 10, 32) or not torch.isfinite(actions).all():
        raise ValueError("official raw actions must be finite with shape (1, 10, 32)")
    raw_actions = actions[0].float().cpu().numpy()
    decoded = policy._output_transform(
        {"state": transformed["state"], "actions": raw_actions.copy()}
    )["actions"]
    if decoded.shape != (10, 7) or not np.isfinite(decoded).all():
        raise ValueError("official decoded actions must be finite with shape (10, 7)")

    metadata = {
        "schema_version": 1,
        "openpi_commit": OPENPI_COMMIT,
        "openpi_source_sha256": source_sha256,
        "config": "pi05_libero",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_size": weight_path.stat().st_size,
        "checkpoint_config_sha256": _sha256(args.checkpoint / "config.json"),
        "norm_stats_sha256": _sha256(stats_path),
        "dataset_root": str(args.dataset_root.resolve()),
        "repo_id": args.repo_id,
        "sample_index": args.sample_index,
        "num_steps": args.num_steps,
        "noise_seed": 7,
    }
    payload = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "raw_state": raw["observation/state"],
        "raw_base_image": raw["observation/image"],
        "raw_wrist_image": raw["observation/wrist_image"],
        "raw_prompt": np.asarray(prompt),
        "noise": noise,
        "state": _numpy(observation.state[0]),
        "tokens": _numpy(observation.tokenized_prompt[0]),
        "token_mask": _numpy(observation.tokenized_prompt_mask[0]),
        "raw_actions": raw_actions,
        "decoded_actions": np.asarray(decoded, dtype=np.float32),
    }
    for key in IMAGE_KEYS:
        payload[f"image_{key}"] = _numpy(observation.images[key][0])
        payload[f"image_mask_{key}"] = _numpy(observation.image_masks[key][0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"golden={args.output}")


if __name__ == "__main__":
    main()

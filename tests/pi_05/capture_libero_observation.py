"""Capture one real LeRobot LIBERO sample as a policy request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="lerobot/libero")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(args.sample_index)
    sample = dataset[args.sample_index]
    task_index = int(sample["task_index"])
    prompt = sample["task"]
    if metadata.fps != 10 or not isinstance(prompt, str):
        raise ValueError("unexpected LIBERO FPS or task prompt")
    state = _numpy(sample["observation.state"]).astype(np.float32)
    base_image = _numpy(sample["observation.images.image"])
    wrist_image = _numpy(sample["observation.images.image2"])
    if state.shape != (8,) or base_image.shape != (3, 256, 256):
        raise ValueError("unexpected LIBERO state or base image shape")
    if wrist_image.shape != base_image.shape or not prompt:
        raise ValueError("unexpected LIBERO wrist image or prompt")

    info = {
        "dataset_root": str(args.dataset_root.resolve()),
        "repo_id": args.repo_id,
        "sample_index": args.sample_index,
        "task_index": task_index,
        "prompt": prompt,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        metadata_json=np.asarray(json.dumps(info, sort_keys=True)),
        raw_state=state,
        raw_base_image=base_image,
        raw_wrist_image=wrist_image,
        raw_prompt=np.asarray(prompt),
    )
    print(json.dumps(info, indent=2, sort_keys=True))
    print(f"observation={args.output}")


if __name__ == "__main__":
    main()

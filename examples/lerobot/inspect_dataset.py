"""Inspect one raw sample from a local LeRobot dataset."""

from __future__ import annotations

import argparse
import pprint
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata


def _preview_tensor(value: torch.Tensor, preview_size: int) -> str:
    flat = value.detach().cpu().reshape(-1)
    preview = flat[:preview_size].tolist()
    value_range = (
        "empty"
        if flat.numel() == 0
        else f"min={flat.min().item()!r}, max={flat.max().item()!r}"
    )
    return (
        f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
        f"{value_range}, preview={preview})"
    )


def _preview_array(value: np.ndarray, preview_size: int) -> str:
    flat = value.reshape(-1)
    preview = flat[:preview_size].tolist()
    value_range = "empty" if flat.size == 0 else f"min={flat.min()!r}, max={flat.max()!r}"
    return f"ndarray(shape={value.shape}, dtype={value.dtype}, {value_range}, preview={preview})"


def _print_value(name: str, value: Any, *, preview_size: int, indent: str = "") -> None:
    if isinstance(value, Mapping):
        print(f"{indent}{name}: dict")
        for key, child in value.items():
            _print_value(str(key), child, preview_size=preview_size, indent=f"{indent}  ")
    elif isinstance(value, torch.Tensor):
        print(f"{indent}{name}: {_preview_tensor(value, preview_size)}")
    elif isinstance(value, np.ndarray):
        print(f"{indent}{name}: {_preview_array(value, preview_size)}")
    else:
        print(f"{indent}{name}: {type(value).__name__}({value!r})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a local LeRobot dataset sample")
    parser.add_argument("--repo-id", default="lerobot/libero")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--preview-size", type=int, default=12)
    args = parser.parse_args()

    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    dataset = LeRobotDataset(args.repo_id, root=args.root)
    assert 0 <= args.index < len(dataset), (
        f"index {args.index} is outside [0, {len(dataset)})"
    )

    print(f"repo_id: {args.repo_id}")
    print(f"root: {args.root}")
    print(f"frames: {len(dataset)}")
    print(f"fps: {metadata.fps}")
    print("\nmetadata.features:")
    pprint.pp(metadata.features, sort_dicts=False)
    print(f"\nsample[{args.index}]:")
    for key, value in dataset[args.index].items():
        _print_value(str(key), value, preview_size=args.preview_size)


if __name__ == "__main__":
    main()

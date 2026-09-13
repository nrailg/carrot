"""Compute PI0.5 quantile statistics from RoboTwin parquet data without decoding video."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
from datasets import load_dataset

from carrot.data.loading import load_callable

Preprocess = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def _update_episode(
    state: np.ndarray,
    action: np.ndarray,
    *,
    action_horizon: int,
    state_stats: Statistics | None,
    action_stats: Statistics | None,
    preprocess: Preprocess | None,
    state_sample_stride: int = 1,
    action_sample_stride: int = 1,
) -> tuple[Statistics, Statistics]:
    """Apply the configured preprocessing to episode-bounded action windows."""
    state_for_stats = state
    if preprocess is not None:
        state_for_stats, _ = preprocess(state, action[:, None])
    if state_stats is None:
        state_stats = Statistics(state_for_stats.shape[-1], sample_stride=state_sample_stride)
    state_stats.update(state_for_stats)
    offsets = np.arange(action_horizon)
    for start in range(0, len(state), 4096):
        stop = min(start + 4096, len(state))
        indices = np.minimum(
            np.arange(start, stop)[:, None] + offsets[None, :],
            len(action) - 1,
        )
        action_windows = action[indices].copy()
        if preprocess is not None:
            _, action_windows = preprocess(state[start:stop], action_windows)
        if action_stats is None:
            action_stats = Statistics(action_windows.shape[-1], sample_stride=action_sample_stride)
        action_stats.update(action_windows)
    if action_stats is None:
        raise ValueError("episode does not contain actions")
    return state_stats, action_stats


class Statistics:
    def __init__(self, dimensions: int, *, sample_stride: int) -> None:
        self.count = 0
        self.sum = np.zeros(dimensions, dtype=np.float64)
        self.square_sum = np.zeros(dimensions, dtype=np.float64)
        self.minimum = np.full(dimensions, np.inf)
        self.maximum = np.full(dimensions, -np.inf)
        self.sample_stride = sample_stride
        self.samples: list[np.ndarray] = []

    def update(self, values: np.ndarray) -> None:
        values = values.reshape(-1, values.shape[-1]).astype(np.float64, copy=False)
        self.sum += values.sum(axis=0)
        self.square_sum += np.square(values).sum(axis=0)
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))
        indices = np.arange(self.count, self.count + len(values))
        sampled = values[indices % self.sample_stride == 0]
        if len(sampled):
            self.samples.append(sampled.astype(np.float32))
        self.count += len(values)

    def result(self) -> dict[str, list[float]]:
        mean = self.sum / self.count
        variance = np.maximum(self.square_sum / self.count - np.square(mean), 0)
        samples = np.concatenate(self.samples)
        return {
            "min": self.minimum.astype(np.float32).tolist(),
            "max": self.maximum.astype(np.float32).tolist(),
            "mean": mean.astype(np.float32).tolist(),
            "std": np.sqrt(variance).astype(np.float32).tolist(),
            "q01": np.quantile(samples, 0.01, axis=0).astype(np.float32).tolist(),
            "q99": np.quantile(samples, 0.99, axis=0).astype(np.float32).tolist(),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--quantile-samples", type=int, default=1_000_000)
    parser.add_argument("--preprocess", default="carrot.data.lerobot.robotwin_preprocess")
    args = parser.parse_args()

    root = Path(args.root)
    with (root / "meta" / "info.json").open() as stream:
        total_frames = int(json.load(stream)["total_frames"])
    state_stride = max(1, math.ceil(total_frames / args.quantile_samples))
    action_stride = max(
        1,
        math.ceil(total_frames * args.action_horizon / args.quantile_samples),
    )
    files = [str(path) for path in sorted((root / "data").rglob("*.parquet"))]
    rows = load_dataset("parquet", data_files=files, split="train", streaming=True)
    state_stats: Statistics | None = None
    action_stats: Statistics | None = None
    preprocess = load_callable(args.preprocess) if args.preprocess else None
    current_episode = None
    state_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []

    def finish_episode() -> None:
        nonlocal state_stats, action_stats
        if not state_chunks:
            return
        state_stats, action_stats = _update_episode(
            np.concatenate(state_chunks),
            np.concatenate(action_chunks),
            action_horizon=args.action_horizon,
            state_stats=state_stats,
            action_stats=action_stats,
            preprocess=preprocess,
            state_sample_stride=state_stride,
            action_sample_stride=action_stride,
        )
        state_chunks.clear()
        action_chunks.clear()

    for batch in rows.iter(batch_size=args.batch_size):
        state = np.asarray(batch["observation.state"], dtype=np.float32)
        action = np.asarray(batch["action"], dtype=np.float32)
        episodes = np.asarray(batch["episode_index"])
        boundaries = np.r_[0, np.flatnonzero(episodes[1:] != episodes[:-1]) + 1, len(episodes)]
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            episode = int(episodes[start])
            if current_episode is not None and episode != current_episode:
                finish_episode()
            current_episode = episode
            state_chunks.append(state[start:stop])
            action_chunks.append(action[start:stop])

    finish_episode()
    if state_stats is None or action_stats is None:
        raise ValueError("dataset does not contain any episodes")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        json.dump(
            {
                "state": state_stats.result(),
                "action": action_stats.result(),
                "meta": {
                    "frames": total_frames,
                    "action_horizon": args.action_horizon,
                },
            },
            stream,
            indent=2,
        )


if __name__ == "__main__":
    main()

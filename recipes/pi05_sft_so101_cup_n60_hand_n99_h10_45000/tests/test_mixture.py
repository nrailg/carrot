"""Offline contracts and real LeRobot v3 H10 samples for the joint SO101 recipe."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import ConcatDataset

from carrot.data.dataset_spec import SFTDatasetSpec
from carrot.data.so101 import SO101SFTDataset
from recipes.pi05_sft_so101_cup_n60_hand_n99_h10_45000 import preflight

MODULE_SPEC = importlib.util.spec_from_file_location(
    "joint_so101_dataset", Path(__file__).resolve().parents[1] / "dataset.py"
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
mixture = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(mixture)

JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
UNITS = ["degree"] * 5 + ["gripper_opening_percent"]
CAMERAS = ("observation.images.top", "observation.images.wrist")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value))


def numeric_stats(values: np.ndarray) -> dict[str, Any]:
    values = values.astype(np.float64)
    return {
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
        "mean": values.mean(0).tolist(),
        "std": values.std(0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
        "count": [len(values)],
    }


def make_source(root: Path, name: str, lengths: tuple[int, ...], offset: int) -> np.ndarray:
    """Write two short episodes and real RGB videos, using ordinary v3 metadata."""
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    features = {
        key: {"dtype": "float32", "shape": [6], "names": JOINTS}
        for key in ("action", "observation.state")
    }
    for camera in CAMERAS:
        features[camera] = {
            "dtype": "video",
            "shape": [16, 16, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": 15,
                "video.codec": "h264",
                "video.height": 16,
                "video.width": 16,
                "video.channels": 3,
                "video.pix_fmt": "yuv420p",
                "has_audio": False,
                "is_depth_map": False,
            },
        }
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        features[key] = {
            "dtype": "float32" if key == "timestamp" else "int64",
            "shape": [1],
            "names": None,
        }
    frames = sum(lengths)
    write_json(
        root / "meta/info.json",
        {
            "codebase_version": "v3.0",
            "fps": 15,
            "features": features,
            "total_frames": frames,
            "total_episodes": len(lengths),
            "total_tasks": len(lengths),
            "robot_type": "so_follower",
            "chunks_size": 1000,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 200,
            "splits": {"train": f"0:{len(lengths)}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        },
    )
    actions = np.arange(frames * 6, dtype=np.float32).reshape(frames, 6) / 10 + offset
    # Constant joints are valid: their standard deviation and quantile range are zero.
    actions[:, 5] = 0.5
    states = actions - np.float32(0.25)
    data = {
        "action": actions.tolist(),
        "observation.state": states.tolist(),
        "timestamp": [i / 15 for length in lengths for i in range(length)],
        "frame_index": [i for length in lengths for i in range(length)],
        "episode_index": [i for i, length in enumerate(lengths) for _ in range(length)],
        "index": list(range(frames)),
        "task_index": [i for i, length in enumerate(lengths) for _ in range(length)],
    }
    schema = pa.schema(
        [
            pa.field(
                key,
                pa.list_(pa.float32(), 6)
                if key in {"action", "observation.state"}
                else pa.float32()
                if key == "timestamp"
                else pa.int64(),
            )
            for key in data
        ]
    )
    pq.write_table(
        pa.Table.from_pydict(data, schema=schema), root / "data/chunk-000/file-000.parquet"
    )
    episodes = []
    start = 0
    for episode, length in enumerate(lengths):
        record = {
            "episode_index": episode,
            "tasks": [f"{name} task {episode}"],
            "length": length,
            "dataset_from_index": start,
            "dataset_to_index": start + length,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for camera in CAMERAS:
            prefix = f"videos/{camera}"
            record.update(
                {
                    prefix + "/chunk_index": 0,
                    prefix + "/file_index": 0,
                    prefix + "/from_timestamp": start / 15,
                    prefix + "/to_timestamp": (start + length) / 15,
                }
            )
        episodes.append(record)
        start += length
    pq.write_table(
        pa.Table.from_pylist(episodes), root / "meta/episodes/chunk-000/file-000.parquet"
    )
    pd.DataFrame(
        {"task_index": list(range(len(lengths)))},
        index=pd.Index([f"{name} task {i}" for i in range(len(lengths))], name="task"),
    ).to_parquet(root / "meta/tasks.parquet")
    stats = {"action": numeric_stats(actions), "observation.state": numeric_stats(states)}
    write_json(root / "meta/stats.json", stats)
    write_json(
        root / "meta/pi05_norm_stats.json",
        {
            "state": stats["observation.state"],
            "actions": stats["action"],
            "meta": {
                "frames": frames,
                "episodes": len(lengths),
                "joint_names": JOINTS,
                "units": UNITS,
            },
        },
    )
    for camera_index, camera in enumerate(CAMERAS):
        path = root / f"videos/{camera}/chunk-000/file-000.mp4"
        path.parent.mkdir(parents=True)
        with av.open(str(path), "w") as container:
            stream = container.add_stream("libx264", rate=15)
            stream.width = stream.height = 16
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "0", "preset": "fast", "bf": "0", "g": "2"}
            for frame_index in range(frames):
                pixels = np.full((16, 16, 3), 30 + camera_index * 100 + frame_index, np.uint8)
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    return actions


@dataclass
class MixtureCase:
    kwargs: dict[str, Any]
    actions: list[np.ndarray]
    lengths: tuple[tuple[int, ...], ...] = ((3, 4), (2, 3))


def make_mixture_case(tmp_path: Path, lengths: tuple[tuple[int, ...], ...]) -> MixtureCase:
    sources, actions = [], []
    for name, source_lengths, offset in zip(("cup", "hand"), lengths, (0, 100), strict=True):
        root = tmp_path / name
        actions.append(make_source(root, name, source_lengths, offset))
        sources.append(
            {
                "repo_id": f"fixture/{name}",
                "root": str(root),
                "revision": "fixture-revision",
                "frames": sum(source_lengths),
                "episodes": len(source_lengths),
            }
        )
    all_actions = np.concatenate(actions)
    norm_path = tmp_path / "joint_norm.json"
    write_json(
        norm_path,
        {
            "state": numeric_stats(all_actions - np.float32(0.25)),
            "actions": numeric_stats(all_actions),
            "meta": {
                "frames": len(all_actions),
                "episodes": sum(map(len, lengths)),
                "joint_names": JOINTS,
                "units": UNITS,
            },
        },
    )
    return MixtureCase(
        {
            "datasets": sources,
            "norm_stats_path": str(norm_path),
            "action_horizon": 10,
            "video_backend": "pyav",
            "base_image_key": CAMERAS[0],
            "wrist_image_key": CAMERAS[1],
        },
        actions,
        lengths,
    )


@pytest.fixture
def mixture_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MixtureCase:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    return make_mixture_case(tmp_path, ((3, 4), (2, 3)))


def test_real_h10_mixture_preserves_frames_prompts_and_episode_padding(mixture_case: MixtureCase):
    spec = mixture.build_dataset(**mixture_case.kwargs)
    assert isinstance(spec, SFTDatasetSpec)
    assert isinstance(spec.dataset, ConcatDataset)
    assert all(isinstance(source, SO101SFTDataset) for source in spec.dataset.datasets)
    assert spec.dataset.cumulative_sizes == [7, 12]
    assert len(spec.dataset) == 12
    assert spec.image_keys == CAMERAS
    assert (spec.state_key, spec.action_key, spec.task_key, spec.embodiment) == (
        "observation/state",
        "actions",
        "prompt",
        "so101",
    )
    norm = json.loads(Path(mixture_case.kwargs["norm_stats_path"]).read_text())
    assert spec.state_stats == norm["state"] and spec.action_stats == norm["actions"]

    # Every source frame appears once. Tail actions repeat the last frame of that
    # episode, including at the source boundary where the next source jumps by 100.
    global_index = 0
    for name, actions, lengths in zip(
        ("cup", "hand"), mixture_case.actions, mixture_case.lengths, strict=True
    ):
        start = 0
        for episode, length in enumerate(lengths):
            for local_index in range(length):
                sample = spec.dataset[global_index]
                expected_indices = start + np.minimum(np.arange(10) + local_index, length - 1)
                np.testing.assert_array_equal(sample["actions"], actions[expected_indices])
                np.testing.assert_array_equal(
                    sample["observation/state"], actions[start + local_index] - np.float32(0.25)
                )
                np.testing.assert_array_equal(
                    sample["action_is_pad"], np.arange(10) + local_index >= length
                )
                assert sample["actions"].dtype == sample["observation/state"].dtype == np.float32
                assert sample["prompt"] == f"{name} task {episode}"
                for key in ("observation/image", "observation/wrist_image"):
                    assert sample[key].shape == (3, 16, 16)
                    assert sample[key].dtype == torch.uint8
                assert sample["observation/image"].float().mean() < 50
                assert sample["observation/wrist_image"].float().mean() > 120
                global_index += 1
            start += length


def test_real_h10_windows_keep_all_ten_actions_before_padding_at_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    case = make_mixture_case(tmp_path, ((14, 3), (12, 2)))
    spec = mixture.build_dataset(**case.kwargs)
    cursor = 0
    for name, actions, lengths in zip(("cup", "hand"), case.actions, case.lengths, strict=True):
        # The first episode provides both an interior H10 window and one ending
        # exactly at its boundary. Steps 5-10 must remain distinct real targets.
        for anchor in (0, lengths[0] - 10):
            sample = spec.dataset[cursor + anchor]
            np.testing.assert_array_equal(sample["actions"], actions[anchor : anchor + 10])
            np.testing.assert_array_equal(sample["action_is_pad"], np.zeros(10, dtype=bool))
            assert np.unique(sample["actions"][4:, 0]).size == 6
            assert sample["prompt"] == f"{name} task 0"

        start = 0
        for episode, length in enumerate(lengths):
            end = start + length
            for anchor in (end - 2, end - 1):
                sample = spec.dataset[cursor + anchor]
                expected_indices = np.minimum(np.arange(anchor, anchor + 10), end - 1)
                np.testing.assert_array_equal(sample["actions"], actions[expected_indices])
                np.testing.assert_array_equal(
                    sample["action_is_pad"], np.arange(anchor, anchor + 10) >= end
                )
                assert sample["prompt"] == f"{name} task {episode}"
            start = end
        cursor += len(actions)

    # The next source starts with its own task and large action offset; the cup
    # source's final padded window above must never consume these hand actions.
    boundary = len(case.actions[0])
    np.testing.assert_array_equal(spec.dataset[boundary]["actions"], case.actions[1][:10])
    assert spec.dataset[boundary - 1]["prompt"] == "cup task 1"
    assert spec.dataset[boundary]["prompt"] == "hand task 0"


@pytest.mark.parametrize("full,expected_samples", [(False, 6), (True, 11)])
def test_preflight_sample_modes_use_real_videos(
    mixture_case: MixtureCase,
    monkeypatch: pytest.MonkeyPatch,
    full: bool,
    expected_samples: int,
) -> None:
    spec = mixture.build_dataset(**mixture_case.kwargs)
    visited = []

    class RecordingDataset:
        def __len__(self):
            return len(spec.dataset)

        def __getitem__(self, index):
            visited.append(index)
            return spec.dataset[index]

    def factory(**kwargs):
        return replace(spec, dataset=RecordingDataset())

    monkeypatch.setattr(preflight, "load_callable", lambda _: factory)
    if not full:
        monkeypatch.setattr(
            preflight.np, "quantile", lambda *args, **kwargs: pytest.fail("Statistics recomputed")
        )
    report = preflight.check_samples(
        {"dataset": {"factory": "fixture", "factory_kwargs": mixture_case.kwargs}},
        Path(mixture_case.kwargs["norm_stats_path"]),
        full=full,
    )
    assert report["frames"] == 12 and report["episodes"] == 4
    assert report["samples_checked"] == len(visited) == expected_samples
    assert report["statistics_recomputed"] is full
    # Check both source boundaries, including the final H10 padded action windows.
    assert {0, 6, 7, 11} <= set(visited)


def test_full_preflight_detects_statistics_that_do_not_match_frame_vectors(
    mixture_case: MixtureCase,
) -> None:
    norm_path = Path(mixture_case.kwargs["norm_stats_path"])
    norm = json.loads(norm_path.read_text())
    norm["state"]["mean"][0] += 0.1  # Valid shape/range, but not the actual frame mean.
    write_json(norm_path, norm)
    config = {
        "dataset": {
            "factory": "recipes.pi05_sft_so101_cup_n60_hand_n99_h10_45000.dataset.build_dataset",
            "factory_kwargs": mixture_case.kwargs,
        }
    }
    # The full acceptance path must retain the independent statistics calculation.
    with pytest.raises(ValueError, match="Joint statistics differ"):
        preflight.check_samples(config, norm_path, full=True)


@pytest.mark.parametrize("fault", ["nonfinite", "action_shape", "padding", "prompt", "camera"])
def test_default_preflight_still_rejects_invalid_adapter_samples(
    mixture_case: MixtureCase, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    spec = mixture.build_dataset(**mixture_case.kwargs)

    class BrokenDataset:
        def __len__(self):
            return len(spec.dataset)

        def __getitem__(self, index):
            sample = dict(spec.dataset[index])
            if fault == "nonfinite":
                sample["actions"] = np.full((10, 6), np.nan, dtype=np.float32)
            elif fault == "action_shape":
                sample["actions"] = sample["actions"][:9]
            elif fault == "padding":
                sample["action_is_pad"] = np.zeros(10, dtype=bool)
            elif fault == "prompt":
                sample["prompt"] = "wrong task"
            else:
                sample["observation/image"] = sample["observation/image"][:, :8, :]
            return sample

    def factory(**kwargs):
        return replace(spec, dataset=BrokenDataset())

    monkeypatch.setattr(preflight, "load_callable", lambda _: factory)
    config = {"dataset": {"factory": "fixture", "factory_kwargs": mixture_case.kwargs}}
    with pytest.raises(ValueError):
        preflight.check_samples(config, Path(mixture_case.kwargs["norm_stats_path"]))


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("datasets", [], "nonempty list"),
        ("datasets", {}, "nonempty list"),
        ("datasets", [None], "must be an object"),
        ("action_horizon", 0, "positive integer"),
        ("action_horizon", 1.5, "positive integer"),
        ("action_horizon", True, "positive integer"),
        ("video_backend", "", "nonempty string"),
        ("base_image_key", CAMERAS[1], "must be distinct"),
        ("base_image_key", "absent", "camera absent"),
    ],
)
def test_invalid_arguments_fail_before_source_loading(
    mixture_case: MixtureCase, monkeypatch: pytest.MonkeyPatch, field: str, value: Any, match: str
):
    monkeypatch.setattr(mixture, "build_source", lambda **_: pytest.fail("Source loading reached"))
    mixture_case.kwargs[field] = value
    with pytest.raises(ValueError, match=match):
        mixture.build_dataset(**mixture_case.kwargs)


@pytest.mark.parametrize(
    "fault,match",
    [
        ("duplicate", "Duplicate dataset roots"),
        ("missing_root", "source root does not exist"),
        ("missing_revision", "revision must be a nonempty string"),
        ("frames", "Unexpected frames count"),
        ("episodes", "Unexpected episodes count"),
        ("invalid_frames", "frames must be a positive integer"),
        ("joints", "joint names or order differ"),
        ("source_norm_order", "Source normalization joint_names"),
        ("units", "source units differs"),
        ("fps", "source fps differs"),
        ("camera", "source cameras differs"),
        ("video_fps", "FPS differs from dataset FPS"),
        ("no_camera", "at least one explicit camera"),
    ],
)
def test_bad_sources_fail_before_loading(
    mixture_case: MixtureCase, monkeypatch: pytest.MonkeyPatch, fault: str, match: str
):
    monkeypatch.setattr(mixture, "build_source", lambda **_: pytest.fail("Source loading reached"))
    kwargs = mixture_case.kwargs
    source = kwargs["datasets"][1]
    root = Path(source["root"])
    info_path, norm_path = root / "meta/info.json", root / "meta/pi05_norm_stats.json"
    info, norm = json.loads(info_path.read_text()), json.loads(norm_path.read_text())
    if fault == "duplicate":
        source["root"] = str(Path(kwargs["datasets"][0]["root"]) / ".." / "cup")
    elif fault == "missing_root":
        source["root"] += "-missing"
    elif fault == "missing_revision":
        del source["revision"]
    elif fault in ("frames", "episodes"):
        source[fault] += 1
    elif fault == "invalid_frames":
        source["frames"] = True
    elif fault == "joints":
        info["features"]["action"]["names"] = JOINTS[::-1]
    elif fault == "source_norm_order":
        norm["meta"]["joint_names"] = JOINTS[::-1]
    elif fault == "units":
        norm["meta"]["units"][0] = "radian"
    elif fault == "fps":
        info["fps"] = 30
        for camera in CAMERAS:
            info["features"][camera]["info"]["video.fps"] = 30
    elif fault == "camera":
        info["features"][CAMERAS[0]]["shape"][0] = 32
    elif fault == "video_fps":
        info["features"][CAMERAS[0]]["info"]["video.fps"] = 30
    elif fault == "no_camera":
        kwargs["base_image_key"] = kwargs["wrist_image_key"] = None
    write_json(info_path, info)
    write_json(norm_path, norm)
    with pytest.raises(ValueError, match=match):
        mixture.build_dataset(**kwargs)


@pytest.mark.parametrize(
    "fault,match",
    [
        ("joint_names", "Shared normalization joint_names"),
        ("units", "Shared normalization units"),
        ("frames", "complete mixture"),
        ("episodes", "complete mixture"),
        ("shape", "six finite numbers"),
        ("nan", "six finite numbers"),
        ("string", "six finite numbers"),
        ("negative_std", "nonnegative"),
        ("quantile_order", "q01 exceeds q99"),
        ("bounds", "outside min/max"),
        ("count", "total frame count"),
    ],
)
def test_invalid_joint_normalization_fails_before_loading(
    mixture_case: MixtureCase, monkeypatch: pytest.MonkeyPatch, fault: str, match: str
):
    monkeypatch.setattr(mixture, "build_source", lambda **_: pytest.fail("Source loading reached"))
    path = Path(mixture_case.kwargs["norm_stats_path"])
    norm = json.loads(path.read_text())
    if fault == "joint_names":
        norm["meta"]["joint_names"] = JOINTS[::-1]
    elif fault == "units":
        norm["meta"]["units"][0] = "radian"
    elif fault in ("frames", "episodes"):
        norm["meta"][fault] += 1
    elif fault == "shape":
        norm["actions"]["mean"].pop()
    elif fault == "nan":
        norm["state"]["q01"][0] = float("nan")
    elif fault == "string":
        norm["state"]["mean"][0] = "1.0"
    elif fault == "negative_std":
        norm["actions"]["std"][0] = -1
    elif fault == "quantile_order":
        norm["state"]["q01"][0] = norm["state"]["q99"][0] + 1
    elif fault == "bounds":
        norm["actions"]["max"][0] = norm["actions"]["min"][0] - 1
    elif fault == "count":
        norm["state"]["count"] = [7]
    write_json(path, norm)
    with pytest.raises(ValueError, match=match):
        mixture.build_dataset(**mixture_case.kwargs)

"""Recipe-local, frame-uniform SO101 mixture with shared joint normalization."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import ConcatDataset

from carrot.data.dataset_spec import SFTDatasetSpec
from carrot.data.so101 import build_dataset as build_source


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Required metadata file does not exist: {path}")
    with path.open() as stream:
        value = json.load(stream)
    return _object(value, str(path))


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _six_names(value: Any, label: str, *, unique: bool = False) -> list[str]:
    if not isinstance(value, list) or len(value) != 6:
        raise ValueError(f"{label} must contain six names in joint order")
    for name in value:
        _text(name, label)
    if unique and len(set(value)) != 6:
        raise ValueError(f"{label} must contain six distinct joint names")
    return value


def _validate_stats(norm: dict[str, Any], frames: int) -> None:
    for key in ("state", "actions"):
        stats = _object(norm.get(key), f"Shared {key} statistics")
        arrays = {}
        for statistic in ("mean", "std", "q01", "q99", "min", "max"):
            if statistic in ("min", "max") and statistic not in stats:
                continue
            values = np.asarray(stats.get(statistic))
            if (
                values.shape != (6,)
                or values.dtype.kind not in "fiu"
                or not np.isfinite(values).all()
            ):
                raise ValueError(f"Shared {key}.{statistic} must be six finite numbers")
            arrays[statistic] = values
        if np.any(arrays["std"] < 0):
            raise ValueError(f"Shared {key}.std must be nonnegative")
        if np.any(arrays["q01"] > arrays["q99"]):
            raise ValueError(f"Shared {key}.q01 exceeds q99")
        if ("min" in arrays) != ("max" in arrays):
            raise ValueError(f"Shared {key} must supply both min and max, or neither")
        if "min" in arrays:
            for statistic in ("mean", "q01", "q99"):
                if np.any(arrays[statistic] < arrays["min"]) or np.any(
                    arrays[statistic] > arrays["max"]
                ):
                    raise ValueError(f"Shared {key}.{statistic} is outside min/max")
        if "count" in stats:
            count = stats["count"]
            if not isinstance(count, list) or len(count) != 1:
                raise ValueError(f"Shared {key}.count must contain the total frame count")
            if _positive_int(count[0], f"Shared {key}.count") != frames:
                raise ValueError(f"Shared {key}.count differs from the total frame count")


def _source_contract(root: Path, item: dict[str, Any], cameras: tuple[str, ...]) -> dict[str, Any]:
    info = _read_object(root / "meta/info.json")
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected LeRobot v3.0 data: {root}")
    for expected, field in (("frames", "total_frames"), ("episodes", "total_episodes")):
        actual = _positive_int(info.get(field), f"{root}: {field}")
        if actual != item[expected]:
            raise ValueError(
                f"Unexpected {expected} count for {root}: {actual} != {item[expected]}"
            )
    fps = info.get("fps")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not np.isfinite(fps)
        or fps < 1
    ):
        raise ValueError(f"SO101 FPS must be finite and at least one: {root}")
    if info.get("robot_type") not in ("so101_follower", "so_follower"):
        raise ValueError(f"Unsupported SO101 robot_type: {root}")
    features = _object(info.get("features"), f"{root}: features")
    joint_features = {}
    for key in ("observation.state", "action"):
        feature = _object(features.get(key), f"{root}: {key}")
        if feature.get("dtype") != "float32" or feature.get("shape") != [6]:
            raise ValueError(f"SO101 {key} must have float32 shape [6]: {root}")
        _six_names(feature.get("names"), f"{root}: {key}.names", unique=True)
        joint_features[key] = feature
    names = joint_features["action"]["names"]
    if names != joint_features["observation.state"]["names"]:
        raise ValueError(f"SO101 state/action joint names or order differ: {root}")
    source_norm = _read_object(root / "meta/pi05_norm_stats.json")
    meta = _object(source_norm.get("meta"), f"{root}: normalization meta")
    if meta.get("joint_names") != names:
        raise ValueError(f"Source normalization joint_names differ from feature order: {root}")
    units = _six_names(meta.get("units"), f"{root}: units")
    camera_contract = {}
    for key in cameras:
        feature = _object(features.get(key), f"{root}: camera {key}")
        shape = feature.get("shape")
        if (
            feature.get("dtype") not in ("video", "image")
            or not isinstance(shape, list)
            or len(shape) != 3
            or shape[-1] != 3
        ):
            raise ValueError(f"SO101 camera {key} must have RGB HWC shape: {root}")
        for dimension in shape:
            _positive_int(dimension, f"{root}: camera {key} shape")
        if feature["dtype"] == "video":
            video_info = _object(feature.get("info"), f"{root}: camera {key} info")
            if video_info.get("video.fps") != fps:
                raise ValueError(f"SO101 camera {key} FPS differs from dataset FPS: {root}")
        camera_contract[key] = {"dtype": feature["dtype"], "shape": shape}
    return {
        "fps": fps,
        "robot_type": info["robot_type"],
        "joint_names": names,
        "units": units,
        "cameras": camera_contract,
    }


def build_dataset(
    *,
    datasets: list[dict[str, Any]],
    norm_stats_path: str,
    action_horizon: int = 10,
    video_backend: str = "pyav",
    base_image_key: str | None = None,
    wrist_image_key: str | None = None,
) -> SFTDatasetSpec:
    """Combine existing local releases without rewriting data or task prompts.

    Every source supplies ``repo_id``, ``root``, ``revision``, ``frames`` and
    ``episodes``. Six absolute joint targets keep the source units and joint order;
    shared statistics must describe all frames in all sources in that same order.
    Sources must agree on joints, units, FPS, robot type and selected RGB cameras.

    The returned core ``SFTDatasetSpec`` wraps a ``ConcatDataset``: ordinary sampling
    is uniform over frame anchors, with no task balancing. The core SO101 adapter
    retains each source's episode-bounded future actions and padding mask.
    No files are written by this factory. Invalid configuration or metadata raises
    ``ValueError`` before loading any source; malformed JSON raises ``JSONDecodeError``.
    """
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("datasets must be a nonempty list of source objects")
    _positive_int(action_horizon, "action_horizon")
    _text(video_backend, "video_backend")
    _text(norm_stats_path, "norm_stats_path")
    cameras = tuple(key for key in (base_image_key, wrist_image_key) if key is not None)
    if not cameras:
        raise ValueError("SO101 requires at least one explicit camera")
    for key in cameras:
        _text(key, "camera key")
    if len(set(cameras)) != len(cameras):
        raise ValueError("Base and wrist camera keys must be distinct")

    roots = []
    for index, item in enumerate(datasets):
        _object(item, f"datasets[{index}]")
        for field in ("repo_id", "root", "revision"):
            _text(item.get(field), f"datasets[{index}].{field}")
        for field in ("frames", "episodes"):
            _positive_int(item.get(field), f"datasets[{index}].{field}")
        if item["frames"] < item["episodes"]:
            raise ValueError(f"datasets[{index}] has fewer frames than episodes")
        root = Path(item["root"]).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"SO101 source root does not exist: {root}")
        if root in roots:
            raise ValueError(f"Duplicate dataset roots would oversample a source: {root}")
        roots.append(root)

    contracts = [
        _source_contract(root, item, cameras) for item, root in zip(datasets, roots, strict=True)
    ]
    reference = contracts[0]
    for root, contract in zip(roots[1:], contracts[1:], strict=True):
        for field, expected in reference.items():
            if contract[field] != expected:
                raise ValueError(f"SO101 source {field} differs: {root}")

    norm = _read_object(Path(norm_stats_path).expanduser())
    meta = _object(norm.get("meta"), "Shared normalization meta")
    totals = {field: sum(item[field] for item in datasets) for field in ("frames", "episodes")}
    for field, expected in totals.items():
        if _positive_int(meta.get(field), f"Shared normalization {field}") != expected:
            raise ValueError(f"Shared normalization {field} does not describe the complete mixture")
    for field in ("joint_names", "units"):
        if meta.get(field) != reference[field]:
            raise ValueError(f"Shared normalization {field} differ from source joint order/units")
    _validate_stats(norm, totals["frames"])

    specs = []
    for item, root in zip(datasets, roots, strict=True):
        spec = build_source(
            repo_id=item["repo_id"],
            root=str(root),
            revision=item["revision"],
            action_horizon=action_horizon,
            video_backend=video_backend,
            base_image_key=base_image_key,
            wrist_image_key=wrist_image_key,
        )
        if len(spec.dataset) != item["frames"]:
            raise ValueError(f"Adapter frame count differs: {root}")
        specs.append(spec)
    return replace(
        specs[0],
        dataset=ConcatDataset([spec.dataset for spec in specs]),
        state_stats=norm["state"],
        action_stats=norm["actions"],
    )

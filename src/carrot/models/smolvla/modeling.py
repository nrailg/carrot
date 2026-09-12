"""Construct SmolVLA from LeRobot 0.6.1 model and dataset factories."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from lerobot.configs import PreTrainedConfig
from lerobot.datasets import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    resolve_delta_timestamps,
)
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.collate import lerobot_collate_fn
from torch import nn

MIN_LEROBOT_VERSION = (0, 6, 1)
PREPROCESSOR_FILENAME = "policy_preprocessor.json"


@dataclass(frozen=True)
class SmolVLAComponents:
    policy: nn.Module
    dataset: Any
    preprocessor: Any
    postprocessor: Any
    collate_fn: Any


def parse_lerobot_version(value: str) -> tuple[int, int, int]:
    """Parse a PEP 440-ish version string into a comparable (major, minor, patch) tuple."""
    core = value.split("+")[0].split(".dev")[0]
    parts = core.split(".")
    numbers = []
    for part in parts[:3]:
        digits = ""
        for char in part:
            if char.isdigit():
                digits += char
            else:
                break
        numbers.append(int(digits or 0))
    while len(numbers) < 3:
        numbers.append(0)
    return numbers[0], numbers[1], numbers[2]


def require_lerobot_version(
    installed: str,
    minimum: tuple[int, int, int] = MIN_LEROBOT_VERSION,
) -> None:
    if parse_lerobot_version(installed) < minimum:
        required = ".".join(str(part) for part in minimum)
        raise ImportError(f"Carrot SFT requires lerobot>={required}, found {installed}")


def has_processor_assets(model_path: str) -> bool:
    """True when processors can be loaded from a local checkpoint or a Hub repo id."""
    path = Path(model_path)
    if path.is_dir():
        return (path / PREPROCESSOR_FILENAME).exists()
    return not path.exists()


def processor_kwargs(
    *,
    model_path: str,
    dataset_stats: Any,
    device: str,
    input_features: dict[str, Any],
    output_features: dict[str, Any],
    normalization_mapping: Any,
    rename_map: dict[str, str] | None = None,
    tokenizer_name: str | None = None,
) -> dict[str, Any]:
    """Build 0.6.1 ``make_pre_post_processors`` kwargs for SFT on a new dataset."""
    overrides: dict[str, Any] = {"device_processor": {"device": device}}
    if rename_map:
        overrides["rename_observations_processor"] = {"rename_map": rename_map}
    if tokenizer_name:
        overrides["tokenizer_processor"] = {"tokenizer_name": tokenizer_name}
    kwargs: dict[str, Any] = {
        "dataset_stats": dataset_stats,
        "preprocessor_overrides": overrides,
    }
    if not has_processor_assets(model_path):
        return kwargs
    kwargs["pretrained_path"] = model_path
    kwargs["preprocessor_overrides"]["normalizer_processor"] = {
        "features": {**input_features, **output_features},
        "norm_map": normalization_mapping,
        "stats": dataset_stats,
    }
    kwargs["postprocessor_overrides"] = {
        "unnormalizer_processor": {
            "features": output_features,
            "norm_map": normalization_mapping,
            "stats": dataset_stats,
        },
    }
    return kwargs


def build_smolvla(
    *,
    model_path: str,
    dataset_repo_id: str,
    dataset_root: str | None,
    device: str,
    video_backend: str | None = None,
    rename_map: dict[str, str] | None = None,
    vlm_path: str,
) -> SmolVLAComponents:
    """Build a fine-tuning policy and data pipeline with LeRobot 0.6.1 factories."""
    require_lerobot_version(version("lerobot"))

    root = Path(dataset_root) if dataset_root is not None else None
    metadata = LeRobotDatasetMetadata(dataset_repo_id, root=root)
    policy_config = PreTrainedConfig.from_pretrained(model_path)
    policy_config.pretrained_path = model_path
    policy_config.device = device
    if policy_config.type != "smolvla":
        raise ValueError(f"expected a SmolVLA checkpoint, got policy type {policy_config.type!r}")

    # Two from_pretrained loads: policy safetensors from model_path, VLM from vlm_path.
    policy_config.vlm_model_name = vlm_path
    policy = make_policy(cfg=policy_config, ds_meta=metadata, rename_map=rename_map)
    delta_timestamps = resolve_delta_timestamps(policy_config, metadata)
    dataset_kwargs: dict[str, Any] = {
        "root": root,
        "delta_timestamps": delta_timestamps,
    }
    if video_backend is not None:
        dataset_kwargs["video_backend"] = video_backend
    dataset = LeRobotDataset(dataset_repo_id, **dataset_kwargs)
    dataset_stats = rename_stats(metadata.stats, rename_map or {})
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        **processor_kwargs(
            model_path=model_path,
            dataset_stats=dataset_stats,
            device=device,
            input_features=policy.config.input_features or {},
            output_features=policy.config.output_features or {},
            normalization_mapping=policy.config.normalization_mapping,
            rename_map=rename_map,
            tokenizer_name=vlm_path,
        ),
    )
    if rename_map:
        inner = preprocessor

        def preprocessor(batch: Any, _inner=inner, _rename_map=rename_map) -> Any:
            renamed = {_rename_map.get(key, key): value for key, value in batch.items()}
            return _inner(renamed)
    collate_fn = lerobot_collate_fn if metadata.has_language_columns else None
    return SmolVLAComponents(
        policy=policy,
        dataset=dataset,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        collate_fn=collate_fn,
    )

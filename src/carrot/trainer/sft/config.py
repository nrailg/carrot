"""Configuration for offline supervised fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from carrot.parallel.config import FSDPConfig


def _from_dict[T](cls: type[T], values: dict[str, Any]) -> T:
    valid = {item.name for item in fields(cls)}
    unknown = set(values) - valid
    assert not unknown, f"unknown {cls.__name__} fields: {sorted(unknown)}"
    return cls(**values)


@dataclass(frozen=True)
class ModelConfig:
    path: str = "Miical/pi05-base"
    tokenizer_path: str = "Miical/pi05-base"

    def __post_init__(self) -> None:
        assert self.path, "model.path cannot be empty"
        assert self.tokenizer_path, "model.tokenizer_path cannot be empty"


@dataclass(frozen=True)
class DatasetConfig:
    factory: str = "carrot.data.robotwin.build_dataset"
    factory_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "repo_id": "lerobot/robotwin_unified",
            "root": None,
            "video_backend": None,
            "action_horizon": 50,
            "image_keys": (
                "observation.images.cam_high",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
            ),
        }
    )
    norm_stats_path: str | None = None
    norm_stats_asset_id: str | None = None
    preprocess: str | None = "carrot.data.robotwin.robotwin_preprocess"
    num_workers: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "factory_kwargs", dict(self.factory_kwargs))
        assert self.factory, "dataset.factory cannot be empty"
        if self.norm_stats_asset_id is not None:
            asset_id = Path(self.norm_stats_asset_id)
            assert (
                not asset_id.is_absolute()
                and self.norm_stats_asset_id
                and ".." not in asset_id.parts
            ), "dataset.norm_stats_asset_id must be a relative asset path"
        assert self.num_workers >= 0, "dataset.num_workers cannot be negative"


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-10
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    warmup_steps: int = 1_000
    decay_steps: int = 30_000
    decay_learning_rate: float = 2.5e-6
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "betas", tuple(self.betas))
        assert not self.learning_rate <= 0, "optimizer.learning_rate must be positive"
        assert not self.weight_decay < 0, "optimizer.weight_decay cannot be negative"
        assert len(self.betas) == 2 and all(0 <= beta < 1 for beta in self.betas), (
            "optimizer.betas must contain two values in [0, 1)"
        )
        assert not self.eps <= 0, "optimizer.eps must be positive"
        assert self.warmup_steps >= 0, "optimizer.warmup_steps cannot be negative"
        assert self.decay_steps >= 1, "optimizer.decay_steps must be positive"
        assert 0 < self.decay_learning_rate <= self.learning_rate, (
            "optimizer.decay_learning_rate must be positive and no greater than learning_rate"
        )
        assert not self.max_grad_norm <= 0, "optimizer.max_grad_norm must be positive"


@dataclass(frozen=True)
class WandBConfig:
    enabled: bool = False
    project: str = "carrot-sft"
    entity: str = "1001"
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity", str(self.entity))
        assert not (self.enabled and not self.project), (
            "wandb.project cannot be empty when wandb is enabled"
        )
        assert not (self.enabled and not self.entity), (
            "wandb.entity cannot be empty when wandb is enabled"
        )


@dataclass(frozen=True)
class SFTConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    output_dir: str = "outputs/pi05_robotwin_sft"
    steps: int = 20_000
    micro_batch_size: int = 1
    global_batch_size: int = 1
    log_freq: int = 10
    save_freq: int = 1_000
    seed: int = 1_000
    dp_size: int = 1
    num_nodes: int = 1

    def __post_init__(self) -> None:
        assert self.steps >= 1, "steps must be positive"
        assert self.micro_batch_size >= 1, "micro_batch_size must be positive"
        assert self.global_batch_size >= 1, "global_batch_size must be positive"
        assert self.log_freq >= 1, "log_freq must be positive"
        assert self.dp_size >= 1, "dp_size must be positive"
        assert self.num_nodes >= 1, "num_nodes must be positive"
        assert self.dp_size % self.num_nodes == 0, "dp_size must be divisible by num_nodes"
        assert self.global_batch_size % (self.micro_batch_size * self.dp_size) == 0, (
            "global_batch_size must be divisible by micro_batch_size * dp_size"
        )
        assert self.save_freq >= 0, "save_freq cannot be negative"
        assert self.output_dir, "output_dir cannot be empty"

    @property
    def gpus_per_node(self) -> int:
        return self.dp_size // self.num_nodes

    @property
    def gas(self) -> int:
        """``global_batch_size / (micro_batch_size * dp_size)``."""
        return self.global_batch_size // (self.micro_batch_size * self.dp_size)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> SFTConfig:
        values = dict(values)
        values["model"] = _from_dict(ModelConfig, values.get("model", {}))
        values["dataset"] = _from_dict(DatasetConfig, values.get("dataset", {}))
        values["optimizer"] = _from_dict(OptimizerConfig, values.get("optimizer", {}))
        values["fsdp"] = _from_dict(FSDPConfig, values.get("fsdp", {}))
        values["wandb"] = _from_dict(WandBConfig, values.get("wandb", {}))
        return _from_dict(cls, values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SFTConfig:
        with Path(path).open() as stream:
            values = yaml.safe_load(stream)
        assert isinstance(values, dict), "SFT config must contain a YAML mapping"
        return cls.from_dict(values)

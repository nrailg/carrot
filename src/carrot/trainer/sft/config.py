"""Configuration for offline supervised fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

from carrot.modeling.config import FSDPConfig

T = TypeVar("T")


def _from_dict(cls: type[T], values: dict[str, Any]) -> T:
    valid = {item.name for item in fields(cls)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


@dataclass(frozen=True)
class ModelConfig:
    path: str = "lerobot/smolvla_base"

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("model.path cannot be empty")


@dataclass(frozen=True)
class DatasetConfig:
    repo_id: str = "lerobot/robotwin_unified"
    root: str | None = None
    video_backend: str | None = None
    num_workers: int = 4

    def __post_init__(self) -> None:
        if not self.repo_id:
            raise ValueError("dataset.repo_id cannot be empty")
        if self.num_workers < 0:
            raise ValueError("dataset.num_workers cannot be negative")


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-10
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    warmup_steps: int = 1_000
    decay_steps: int = 30_000
    decay_learning_rate: float = 2.5e-6
    max_grad_norm: float = 10.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "betas", tuple(self.betas))
        if self.learning_rate <= 0:
            raise ValueError("optimizer.learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("optimizer.weight_decay cannot be negative")
        if len(self.betas) != 2 or any(not 0 <= beta < 1 for beta in self.betas):
            raise ValueError("optimizer.betas must contain two values in [0, 1)")
        if self.eps <= 0:
            raise ValueError("optimizer.eps must be positive")
        if self.warmup_steps < 0:
            raise ValueError("optimizer.warmup_steps cannot be negative")
        if self.decay_steps < 1:
            raise ValueError("optimizer.decay_steps must be positive")
        if not 0 < self.decay_learning_rate <= self.learning_rate:
            raise ValueError(
                "optimizer.decay_learning_rate must be positive and no greater than learning_rate"
            )
        if self.max_grad_norm < 0:
            raise ValueError("optimizer.max_grad_norm cannot be negative")


@dataclass(frozen=True)
class SFTConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    output_dir: str = "outputs/smolvla_robotwin_sft"
    steps: int = 20_000
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    log_freq: int = 10
    save_freq: int = 1_000
    seed: int = 1_000
    num_gpus: int = 1

    def __post_init__(self) -> None:
        positive = ("steps", "batch_size", "gradient_accumulation_steps", "log_freq", "num_gpus")
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.save_freq < 0:
            raise ValueError("save_freq cannot be negative")
        if not self.output_dir:
            raise ValueError("output_dir cannot be empty")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> SFTConfig:
        values = dict(values)
        values["model"] = _from_dict(ModelConfig, values.get("model", {}))
        values["dataset"] = _from_dict(DatasetConfig, values.get("dataset", {}))
        values["optimizer"] = _from_dict(OptimizerConfig, values.get("optimizer", {}))
        values["fsdp"] = _from_dict(FSDPConfig, values.get("fsdp", {}))
        return _from_dict(cls, values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SFTConfig:
        try:
            import yaml
        except ImportError as error:
            raise ImportError("YAML configs require PyYAML") from error
        with Path(path).open() as stream:
            values = yaml.safe_load(stream)
        if not isinstance(values, dict):
            raise ValueError("SFT config must contain a YAML mapping")
        return cls.from_dict(values)

"""Configuration for composable model parallelism."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class FSDPConfig:
    enabled: bool = True
    param_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    reduce_dtype: Literal["bfloat16", "float32"] = "float32"
    reshard_after_forward: bool = True
    forward_prefetch: int = 0
    backward_prefetch: int = 0

    def __post_init__(self) -> None:
        assert self.param_dtype in {"bfloat16", "float32"}, (
            f"unsupported param_dtype {self.param_dtype!r}"
        )
        assert self.reduce_dtype in {"bfloat16", "float32"}, (
            f"unsupported reduce_dtype {self.reduce_dtype!r}"
        )
        assert self.forward_prefetch >= 0, "forward_prefetch cannot be negative"
        assert self.backward_prefetch >= 0, "backward_prefetch cannot be negative"

"""PI0.5 inference transforms using the training preprocessing implementation."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from carrot.models.pi05.preprocessing import Pi05Preprocessor


class DataTransformFn(Protocol):
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class CompositeTransform:
    transforms: Sequence[DataTransformFn]

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> CompositeTransform:
    return CompositeTransform(transforms)


@dataclass(frozen=True)
class Normalize:
    norm_stats: dict[str, dict[str, Any]]

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            **data,
            **{
                key: Pi05Preprocessor._normalize(data[key], stats)
                for key, stats in self.norm_stats.items()
                if key in data
            },
        }


@dataclass(frozen=True)
class Unnormalize:
    norm_stats: dict[str, dict[str, Any]]

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        result = dict(data)
        for key, stats in self.norm_stats.items():
            x = data[key]
            q01 = torch.as_tensor(stats["q01"], device=x.device, dtype=torch.float32)
            q99 = torch.as_tensor(stats["q99"], device=x.device, dtype=torch.float32)
            width = q01.shape[-1]
            head = (x[..., :width].float() + 1) / 2 * (q99 - q01 + 1e-6) + q01
            result[key] = torch.cat((head, x[..., width:].float()), dim=-1)
        return result


@dataclass(frozen=True)
class AbsoluteActions:
    mask: tuple[bool, ...] = (True,) * 6 + (False,) + (True,) * 6 + (False,)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        actions = data["actions"].clone()
        # 整个 chunk 的 delta 都相对于推理时的 state，而不是前一个预测动作。
        actions[..., :14] += torch.where(
            torch.tensor(self.mask, device=actions.device), data["state"][..., :14], 0
        ).unsqueeze(-2)
        return {**data, "actions": actions}

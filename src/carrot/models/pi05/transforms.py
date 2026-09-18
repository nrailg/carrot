"""Composable transforms shared by PI0.5 training and inference."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F

from .preprocessing import Pi05Preprocessor


class DataTransformFn(Protocol):
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Pi05TransformSpec:
    """Define model input/output transforms for one embodiment.

    Parameters
    ----------
    inputs : tuple[DataTransformFn, ...]
        Raw environment or repacked dataset sample to canonical model fields.
    outputs : tuple[DataTransformFn, ...]
        Padded model output to environment actions.
    action_dim : int
        Action width returned by the output pipeline.
    """

    inputs: tuple[DataTransformFn, ...]
    outputs: tuple[DataTransformFn, ...]
    action_dim: int


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
class InjectDefaultPrompt:
    prompt: str | None

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        prompt = data.get("prompt", self.prompt)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("a non-empty prompt or default_prompt is required")
        return {**data, "prompt": prompt}


@dataclass(frozen=True)
class Normalize:
    norm_stats: dict[str, dict[str, Any]]
    use_quantiles: bool
    dimensions: dict[str, int]

    def __post_init__(self) -> None:
        first_name, second_name = ("q01", "q99") if self.use_quantiles else ("mean", "std")
        for key, width in self.dimensions.items():
            if key not in self.norm_stats:
                raise ValueError(f"normalization stats missing {key}")
            first = np.asarray(self.norm_stats[key][first_name], dtype=np.float32)
            second = np.asarray(self.norm_stats[key][second_name], dtype=np.float32)
            if first.shape != (width,) or second.shape != (width,):
                label = "quantiles" if self.use_quantiles else "mean/std"
                raise ValueError(f"{key} {label} must have shape ({width},)")
            if not np.isfinite(first).all() or not np.isfinite(second).all():
                label = "quantiles" if self.use_quantiles else "mean/std"
                raise ValueError(f"{key} {label} must be finite")
            if self.use_quantiles and (second < first).any():
                raise ValueError(f"{key} quantiles must be ordered")
            if not self.use_quantiles and (second < 0).any():
                raise ValueError(f"{key} standard deviation must be non-negative")

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        result = dict(data)
        for key, stats in self.norm_stats.items():
            if key not in data:
                continue
            result[key] = _normalize(data[key], stats, use_quantiles=self.use_quantiles)
        return result


@dataclass(frozen=True)
class Unnormalize:
    norm_stats: dict[str, dict[str, Any]]
    use_quantiles: bool

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        result = dict(data)
        for key, stats in self.norm_stats.items():
            if key in data:
                result[key] = _unnormalize(data[key], stats, use_quantiles=self.use_quantiles)
        return result


@dataclass(frozen=True)
class PadStatesAndActions:
    action_dim: int

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        result = {**data, "state": _pad_last(data["state"], self.action_dim)}
        if "actions" in data:
            result["actions"] = _pad_last(data["actions"], self.action_dim)
        return result


@dataclass(frozen=True)
class ResizeImages:
    height: int = 224
    width: int = 224

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        images = {}
        for key, value in data["image"].items():
            image = torch.as_tensor(value)
            unbatched = image.ndim == 3
            if unbatched:
                image = image[None]
            if image.ndim != 4 or image.shape[1] != 3:
                raise ValueError(f"{key} must have shape (3, H, W) or (B, 3, H, W)")
            if not torch.isfinite(image.float()).all():
                raise ValueError(f"{key} must be finite")
            scale_from_uint8 = not image.is_floating_point()
            image = image.float()
            if scale_from_uint8 or image.max() > 1:
                image = image / 255
            current_height, current_width = image.shape[-2:]
            if (current_height, current_width) != (self.height, self.width):
                ratio = max(current_width / self.width, current_height / self.height)
                resized_height = int(current_height / ratio)
                resized_width = int(current_width / ratio)
                image = F.interpolate(
                    image,
                    size=(resized_height, resized_width),
                    mode="bilinear",
                    align_corners=False,
                )
                pad_height = self.height - resized_height
                pad_width = self.width - resized_width
                image = F.pad(
                    image,
                    (
                        pad_width // 2,
                        pad_width - pad_width // 2,
                        pad_height // 2,
                        pad_height - pad_height // 2,
                    ),
                )
            image = 2 * image - 1
            images[key] = image[0] if unbatched else image
        return {**data, "image": images}


class TokenizePrompt(Pi05Preprocessor):
    """Tokenize task text using the PI0 or PI0.5 discrete-state prompt contract.

    Parameters
    ----------
    tokenizer : Any
    discrete_state_input : bool
    """

    def __init__(self, tokenizer: Any, *, discrete_state_input: bool) -> None:
        super().__init__(tokenizer)
        self.discrete_state_input = discrete_state_input

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        state = torch.as_tensor(data["state"])
        unbatched = state.ndim == 1
        if unbatched:
            state = state[None]
        tokens, masks = self._tokenize(
            data["prompt"], state, discrete_state_input=self.discrete_state_input
        )
        if unbatched:
            tokens = tokens[0]
            masks = masks[0]
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": masks,
        }


@dataclass(frozen=True)
class AbsoluteActions:
    mask: tuple[bool, ...] = (True,) * 6 + (False,) + (True,) * 6 + (False,)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        actions = data["actions"].clone()
        actions[..., :14] += torch.where(
            torch.tensor(self.mask, device=actions.device), data["state"][..., :14], 0
        ).unsqueeze(-2)
        return {**data, "actions": actions}


def _normalize(value: Any, stats: dict[str, Any], *, use_quantiles: bool) -> Any:
    first_name, second_name = ("q01", "q99") if use_quantiles else ("mean", "std")
    if isinstance(value, torch.Tensor):
        first = torch.as_tensor(stats[first_name], device=value.device, dtype=torch.float32)
        second = torch.as_tensor(stats[second_name], device=value.device, dtype=torch.float32)
        width = min(value.shape[-1], first.shape[-1])
        if use_quantiles:
            head = 2 * (value[..., :width].float() - first[:width]) / (
                second[:width] - first[:width] + 1e-6
            ) - 1
        else:
            head = (value[..., :width].float() - first[:width]) / (second[:width] + 1e-6)
        return torch.cat((head, value[..., width:].float()), dim=-1)
    array = np.asarray(value, dtype=np.float32)
    first = np.asarray(stats[first_name], dtype=np.float32)
    second = np.asarray(stats[second_name], dtype=np.float32)
    width = min(array.shape[-1], first.shape[-1])
    if use_quantiles:
        head = 2 * (array[..., :width] - first[:width]) / (
            second[:width] - first[:width] + 1e-6
        ) - 1
    else:
        head = (array[..., :width] - first[:width]) / (second[:width] + 1e-6)
    return np.concatenate((head, array[..., width:]), axis=-1)


def _unnormalize(value: Any, stats: dict[str, Any], *, use_quantiles: bool) -> Any:
    first_name, second_name = ("q01", "q99") if use_quantiles else ("mean", "std")
    if isinstance(value, torch.Tensor):
        first = torch.as_tensor(stats[first_name], device=value.device, dtype=torch.float32)
        second = torch.as_tensor(stats[second_name], device=value.device, dtype=torch.float32)
        width = first.shape[-1]
        if use_quantiles:
            head = (value[..., :width].float() + 1) / 2 * (second - first + 1e-6) + first
        else:
            head = value[..., :width].float() * (second + 1e-6) + first
        return torch.cat((head, value[..., width:].float()), dim=-1)
    array = np.asarray(value, dtype=np.float32)
    first = np.asarray(stats[first_name], dtype=np.float32)
    second = np.asarray(stats[second_name], dtype=np.float32)
    width = first.shape[-1]
    if use_quantiles:
        head = (array[..., :width] + 1) / 2 * (second - first + 1e-6) + first
    else:
        head = array[..., :width] * (second + 1e-6) + first
    return np.concatenate((head, array[..., width:]), axis=-1)


def _pad_last(value: Any, width: int) -> Any:
    if value.shape[-1] >= width:
        return value[..., :width]
    if isinstance(value, torch.Tensor):
        return F.pad(value, (0, width - value.shape[-1]))
    return np.pad(value, [(0, 0)] * (value.ndim - 1) + [(0, width - value.shape[-1])])

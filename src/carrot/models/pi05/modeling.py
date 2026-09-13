"""PI0.5 SFT policy and LeRobot dataset adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from carrot.data import SFTDatasetSpec
from carrot.data.loading import load_callable

from .model import PI0Policy


@dataclass(frozen=True)
class Pi05Components:
    policy: PI0Policy
    loss_fn: Pi05SFTLossFn
    dataset: Any
    collate_fn: Any


def _stats_tensor(stats: dict[str, Any], name: str, device: torch.device) -> torch.Tensor:
    value = stats[name]
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, device=device, dtype=torch.float32)


class Pi05SFTLossFn:
    """Compute PI0.5 SFT loss from a batch and native policy."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        state_stats: dict[str, Any],
        action_stats: dict[str, Any],
        image_keys: tuple[str, str, str],
        state_key: str = "observation.state",
        action_key: str = "action",
        task_key: str = "task",
        preprocess: Any | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.state_stats = state_stats
        self.action_stats = action_stats
        self.image_keys = image_keys
        self.state_key = state_key
        self.action_key = action_key
        self.task_key = task_key
        self.preprocess = preprocess

    @staticmethod
    def _normalize(x: torch.Tensor, stats: dict[str, Any]) -> torch.Tensor:
        lower = "q01" if "q01" in stats else "min"
        upper = "q99" if "q99" in stats else "max"
        q01 = _stats_tensor(stats, lower, x.device)
        q99 = _stats_tensor(stats, upper, x.device)
        width = min(x.shape[-1], q01.shape[-1])
        head = 2 * (x[..., :width].float() - q01[:width]) / (q99[:width] - q01[:width] + 1e-6) - 1
        return torch.cat((head, x[..., width:].float()), dim=-1)

    @staticmethod
    def _pad_last(x: torch.Tensor, width: int) -> torch.Tensor:
        return F.pad(x, (0, width - x.shape[-1])) if x.shape[-1] < width else x[..., :width]

    @staticmethod
    def _prepare_image(image: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        image = image.float()
        if image.max() > 1:
            image = image / 255
        if image.shape[-2:] != (224, 224):
            height, width = image.shape[-2:]
            ratio = max(width / 224, height / 224)
            resized_height = int(height / ratio)
            resized_width = int(width / ratio)
            image = F.interpolate(
                image,
                size=(resized_height, resized_width),
                mode="bilinear",
                align_corners=False,
            )
            pad_height = 224 - resized_height
            pad_width = 224 - resized_width
            image = F.pad(
                image,
                (
                    pad_width // 2,
                    pad_width - pad_width // 2,
                    pad_height // 2,
                    pad_height - pad_height // 2,
                ),
            )
        return (2 * image - 1).to(dtype)

    def _tokenize(self, tasks: Any, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(tasks, str):
            tasks = [tasks]
        elif hasattr(tasks, "tolist"):
            tasks = tasks.tolist()
        bins = torch.linspace(-1, 1, 257, device=state.device)[:-1]
        discrete = torch.bucketize(state, bins) - 1
        prompts = []
        for task, values in zip(tasks, discrete, strict=True):
            state_text = " ".join(str(int(value)) for value in values)
            clean_task = str(task).strip().replace("_", " ").replace("\n", " ")
            prompts.append(f"Task: {clean_task}, State: {state_text};\nAction: ")
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            padding_side="right",
            truncation=True,
            max_length=200,
            return_tensors="pt",
        )
        return (
            tokens["input_ids"].to(device=state.device, dtype=torch.long),
            tokens["attention_mask"].to(device=state.device, dtype=torch.bool),
        )

    def __call__(
        self, policy: PI0Policy, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        device = next(policy.parameters()).device
        state = batch[self.state_key].to(device, non_blocking=True)
        actions = batch[self.action_key].to(device, non_blocking=True)
        if self.preprocess is not None:
            state, actions = self.preprocess(state, actions)
        state = self._pad_last(self._normalize(state, self.state_stats), policy.max_state_dim)
        actions = self._pad_last(
            self._normalize(actions, self.action_stats), policy.max_action_dim
        )
        dtype = policy.action_in_proj.weight.dtype
        images = [
            self._prepare_image(batch[key].to(device, non_blocking=True), dtype)
            for key in self.image_keys
        ]
        masks = [torch.ones(images[0].shape[0], device=device, dtype=torch.bool) for _ in images]
        lang_tokens, lang_masks = self._tokenize(batch[self.task_key], state)

        actions = actions.to(dtype)
        noise = torch.randn_like(actions)
        gamma1 = torch.rand(actions.shape[0], device=device).pow(1 / 1.5)
        gamma2 = torch.rand(actions.shape[0], device=device)
        time = (gamma1 / (gamma1 + gamma2) * 0.999 + 0.001).to(dtype)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        target = noise - actions
        prediction = policy(images, masks, lang_tokens, lang_masks, state.to(dtype), x_t, time)
        per_step = torch.square(prediction - target).mean(dim=-1)
        valid = ~batch.get("action_is_pad", torch.zeros_like(per_step, dtype=torch.bool)).to(device)
        loss = (per_step * valid).sum() / valid.sum()
        return loss, {"loss": loss.detach(), "per_step_loss": per_step.detach()}

    def save_artifacts(self, path: Path) -> None:
        """Write non-model files needed by a PI0.5 export.

        Parameters
        ----------
        path : pathlib.Path
            Existing HuggingFace export directory.
        """
        self.tokenizer.save_pretrained(path)
        serializable_stats = {
            "state": {
                key: torch.as_tensor(value).tolist()
                for key, value in self.state_stats.items()
            },
            "action": {
                key: torch.as_tensor(value).tolist()
                for key, value in self.action_stats.items()
            },
        }
        with (Path(path) / "norm_stats.json").open("w") as stream:
            json.dump(serializable_stats, stream)


def build_pi05(
    *,
    model_path: str,
    tokenizer_path: str,
    dataset_factory: str,
    dataset_factory_kwargs: dict[str, Any],
    device: str,
    norm_stats_path: str | None = None,
    preprocess: str | None = "carrot.data.lerobot.robotwin_preprocess",
) -> Pi05Components:
    """Load PI0.5 and bind a configured SFT dataset integration."""
    dataset = load_callable(dataset_factory)(**dataset_factory_kwargs)
    if not isinstance(dataset, SFTDatasetSpec):
        raise TypeError(f"dataset factory {dataset_factory!r} must return SFTDatasetSpec")
    checkpoint_stats = Path(model_path) / "norm_stats.json"
    if norm_stats_path is not None:
        with Path(norm_stats_path).open() as stream:
            normalization = json.load(stream)
        state_stats = normalization["state"]
        action_stats = normalization["action"]
    elif checkpoint_stats.is_file():
        with checkpoint_stats.open() as stream:
            normalization = json.load(stream)
        state_stats = normalization["state"]
        action_stats = normalization["action"]
    else:
        state_stats = dataset.state_stats
        action_stats = dataset.action_stats
    policy = PI0Policy.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, fix_mistral_regex=True)
    loss_fn = Pi05SFTLossFn(
        tokenizer,
        state_stats=state_stats,
        action_stats=action_stats,
        image_keys=dataset.image_keys,
        state_key=dataset.state_key,
        action_key=dataset.action_key,
        task_key=dataset.task_key,
        preprocess=load_callable(preprocess) if preprocess is not None else None,
    )
    return Pi05Components(
        policy=policy.to(device),
        loss_fn=loss_fn,
        dataset=dataset.dataset,
        collate_fn=dataset.collate_fn,
    )

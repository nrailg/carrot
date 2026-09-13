"""PI0.5 SFT policy and LeRobot dataset adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.collate import lerobot_collate_fn
from torch import nn
from transformers import AutoTokenizer

from .model import PI0Policy


@dataclass(frozen=True)
class Pi05Components:
    policy: nn.Module
    dataset: Any
    collate_fn: Any


def _stats_tensor(stats: dict[str, Any], name: str, device: torch.device) -> torch.Tensor:
    value = stats[name]
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, device=device, dtype=torch.float32)


class Pi05SFTPolicy(nn.Module):
    """Own the complete batch-to-loss contract around the native PI0.5 model."""

    def __init__(
        self,
        policy: PI0Policy,
        tokenizer: Any,
        *,
        state_stats: dict[str, Any],
        action_stats: dict[str, Any],
        image_keys: tuple[str, str, str],
        state_key: str = "observation.state",
        action_key: str = "action",
        task_key: str = "task",
        adapt_aloha: bool = True,
        delta_actions: bool = True,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.tokenizer = tokenizer
        self.state_stats = state_stats
        self.action_stats = action_stats
        self.image_keys = image_keys
        self.state_key = state_key
        self.action_key = action_key
        self.task_key = task_key
        self.adapt_aloha = adapt_aloha
        self.delta_actions = delta_actions

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

    @staticmethod
    def _adapt_aloha(
        state: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flip = state.new_tensor([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])
        state = state.clone()
        actions = actions.clone()
        state[..., :14] *= flip
        actions[..., :14] *= flip
        linear = 0.01844 + state[..., [6, 13]] * (0.05800 - 0.01844)
        ratio = (0.022**2 + linear**2 - 0.036**2) / (2 * 0.022 * linear)
        radians = torch.asin(torch.clamp(ratio, -1, 1))
        state[..., [6, 13]] = (radians - 0.5476) / (1.6296 - 0.5476)
        actions[..., [6, 13]] = (
            -0.6213 + actions[..., [6, 13]] * (1.4910 + 0.6213)
        ) - 0.5476
        return state, actions

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

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        device = next(self.policy.parameters()).device
        state = batch[self.state_key].to(device, non_blocking=True)
        actions = batch[self.action_key].to(device, non_blocking=True)
        if self.adapt_aloha:
            state, actions = self._adapt_aloha(state, actions)
        if self.delta_actions:
            mask = state.new_tensor([True] * 6 + [False] + [True] * 6 + [False], dtype=torch.bool)
            actions = actions.clone()
            actions[..., mask] -= state[..., mask].unsqueeze(-2)
        state = self._pad_last(self._normalize(state, self.state_stats), self.policy.max_state_dim)
        actions = self._pad_last(
            self._normalize(actions, self.action_stats), self.policy.max_action_dim
        )
        dtype = self.policy.action_in_proj.weight.dtype
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
        prediction = self.policy(images, masks, lang_tokens, lang_masks, state.to(dtype), x_t, time)
        per_step = torch.square(prediction - target).mean(dim=-1)
        valid = ~batch.get("action_is_pad", torch.zeros_like(per_step, dtype=torch.bool)).to(device)
        loss = (per_step * valid).sum() / valid.sum()
        return loss, {"loss": loss.detach(), "per_step_loss": per_step.detach()}

    def save_pretrained(
        self, path: str | Path, *, state_dict: dict[str, Any] | None = None
    ) -> None:
        if state_dict is not None:
            state_dict = {
                key.removeprefix("policy."): value for key, value in state_dict.items()
            }
        self.policy.save_pretrained(path, state_dict=state_dict, safe_serialization=True)
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
    dataset_repo_id: str,
    dataset_root: str | None,
    image_keys: tuple[str, str, str],
    device: str,
    video_backend: str | None = None,
    norm_stats_path: str | None = None,
    adapt_aloha: bool = True,
    delta_actions: bool = True,
) -> Pi05Components:
    """Load PI0.5, construct 50-step RobotWin samples, and bind the SFT objective."""
    root = Path(dataset_root) if dataset_root else None
    metadata = LeRobotDatasetMetadata(dataset_repo_id, root=root)
    delta_timestamps = {"action": [index / metadata.fps for index in range(50)]}
    dataset_kwargs: dict[str, Any] = {"root": root, "delta_timestamps": delta_timestamps}
    if video_backend is not None:
        dataset_kwargs["video_backend"] = video_backend
    dataset = LeRobotDataset(dataset_repo_id, **dataset_kwargs)
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
        state_stats = metadata.stats["observation.state"]
        action_stats = metadata.stats["action"]
    policy = PI0Policy.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, fix_mistral_regex=True)
    wrapped = Pi05SFTPolicy(
        policy,
        tokenizer,
        state_stats=state_stats,
        action_stats=action_stats,
        image_keys=image_keys,
        adapt_aloha=adapt_aloha,
        delta_actions=delta_actions,
    ).to(device)
    return Pi05Components(
        policy=wrapped,
        dataset=dataset,
        collate_fn=lerobot_collate_fn if metadata.has_language_columns else None,
    )

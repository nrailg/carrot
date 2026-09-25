"""PI0.5 SFT model and LeRobot dataset adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from carrot.data import SFTDatasetSpec
from carrot.data.loading import load_callable

from .embodiments.libero import create_libero_transform_spec
from .embodiments.so101 import create_so101_transform_spec
from .model import PI0Observation, PI0Pytorch
from .preprocessing import Pi05Preprocessor
from .transforms import Pi05TransformSpec, compose, to_observation


@dataclass(frozen=True)
class Pi05Components:
    model: PI0Pytorch
    loss_fn: Pi05SFTLossFn
    dataset: Any
    collate_fn: Any


class Pi05TransformedDataset(Dataset[dict[str, Any]]):
    """Apply an embodiment's inference input transforms before batch collation."""

    def __init__(self, source: Dataset[dict[str, Any]], transform_spec: Pi05TransformSpec) -> None:
        self.source = source
        self.input_transform = compose(transform_spec.inputs)

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw = self.source[index]
        transformed = self.input_transform(raw)
        return {
            "image": transformed["image"],
            "image_mask": transformed["image_mask"],
            "state": transformed["state"],
            "tokenized_prompt": transformed["tokenized_prompt"],
            "tokenized_prompt_mask": transformed["tokenized_prompt_mask"],
            "actions": transformed["actions"],
            "action_is_pad": raw["action_is_pad"],
        }


class Pi05SFTLossFn(Pi05Preprocessor):
    """Compute PI0.5 SFT loss from a batch and native model."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        state_stats: dict[str, Any],
        action_stats: dict[str, Any],
        image_keys: tuple[str, ...],
        state_key: str = "observation.state",
        action_key: str = "action",
        task_key: str = "task",
        preprocess: Any | None = None,
        transform_spec: Pi05TransformSpec | None = None,
    ) -> None:
        super().__init__(tokenizer)
        self.state_stats = state_stats
        self.action_stats = action_stats
        self.image_keys = image_keys
        self.state_key = state_key
        self.action_key = action_key
        self.task_key = task_key
        self.preprocess = preprocess
        self.transform_spec = transform_spec

    def __call__(
        self, model: PI0Pytorch, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        observation, actions, valid = self.prepare_inputs(model, batch)
        device = actions.device
        dtype = actions.dtype
        noise = torch.randn_like(actions)
        gamma1 = torch.rand(actions.shape[0], device=device).pow(1 / 1.5)
        gamma2 = torch.rand(actions.shape[0], device=device)
        time = (gamma1 / (gamma1 + gamma2) * 0.999 + 0.001).to(dtype)
        per_step = model(observation, actions, noise=noise, time=time).mean(dim=-1)
        loss = (per_step * valid).sum() / valid.sum()
        return loss, {"loss": loss.detach(), "per_step_loss": per_step.detach()}

    def prepare_inputs(
        self, model: PI0Pytorch, batch: dict[str, Any]
    ) -> tuple[PI0Observation, torch.Tensor, torch.Tensor]:
        """Construct the model input, normalized target, and valid-step mask.

        Parameters
        ----------
        model : PI0Pytorch
        batch : dict
            Raw RobotWin batch or pre-transformed LIBERO/SO101 batch.

        Returns
        -------
        tuple[PI0Observation, torch.Tensor, torch.Tensor]
            Batched observation, padded actions, and boolean valid-step mask.
        """
        device = next(model.parameters()).device
        dtype = model.action_in_proj.weight.dtype
        if self.transform_spec is not None:
            observation = to_observation(batch, device=device, dtype=dtype)
            actions = torch.as_tensor(batch["actions"], device=device, dtype=dtype)
            valid = ~torch.as_tensor(batch["action_is_pad"], device=device, dtype=torch.bool)
            expected = (
                observation.state.shape[0],
                model.config.action_horizon,
                model.config.action_dim,
            )
            if tuple(actions.shape) != expected or not torch.isfinite(actions).all():
                raise ValueError(f"transformed actions must be finite with shape {expected}")
            if tuple(valid.shape) != expected[:2]:
                raise ValueError(f"action_is_pad must have shape {expected[:2]}")
            return observation, actions, valid

        state = batch[self.state_key].to(device, non_blocking=True)
        actions = batch[self.action_key].to(device, non_blocking=True)
        if self.preprocess is not None:
            state, actions = self.preprocess(state, actions)
        state = self._pad_last(
            self._normalize(state, self.state_stats), model.config.action_dim
        )
        actions = self._pad_last(
            self._normalize(actions, self.action_stats), model.config.action_dim
        )
        images = [
            self._prepare_image(batch[key].to(device, non_blocking=True), dtype)
            for key in self.image_keys
        ]
        masks = [torch.ones(images[0].shape[0], device=device, dtype=torch.bool) for _ in images]
        lang_tokens, lang_masks = self._tokenize(batch[self.task_key], state)

        observation = PI0Observation(
            images=dict(
                zip(("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"), images, strict=True)
            ),
            image_masks=dict(
                zip(("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"), masks, strict=True)
            ),
            state=state.to(dtype),
            tokenized_prompt=lang_tokens,
            tokenized_prompt_mask=lang_masks,
        )
        actions = actions.to(dtype)
        valid = ~batch.get(
            "action_is_pad", torch.zeros(actions.shape[:2], dtype=torch.bool, device=device)
        ).to(device)
        return observation, actions, valid

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
    preprocess: str | None = None,
) -> Pi05Components:
    """Load PI0.5 and bind a configured SFT dataset integration."""
    dataset = load_callable(dataset_factory)(**dataset_factory_kwargs)
    if not isinstance(dataset, SFTDatasetSpec):
        raise TypeError(f"dataset factory {dataset_factory!r} must return SFTDatasetSpec")
    checkpoint_stats = Path(model_path) / "norm_stats.json"
    if dataset.embodiment == "so101":
        if preprocess is not None:
            raise ValueError("SO101 uses shared input transforms; set dataset.preprocess to null")
        if norm_stats_path is None:
            state_stats = dataset.state_stats
            action_stats = dataset.action_stats
        else:
            with Path(norm_stats_path).open() as stream:
                normalization = json.load(stream)
            normalization = normalization.get("norm_stats", normalization)
            state_stats = normalization["state"]
            action_stats = normalization["actions" if "actions" in normalization else "action"]
    elif dataset.embodiment == "libero":
        if preprocess is not None:
            raise ValueError("LIBERO uses shared input transforms; set dataset.preprocess to null")
        official_stats = Path(model_path) / "assets/physical-intelligence/libero/norm_stats.json"
        stats_path = (
            Path(norm_stats_path)
            if norm_stats_path is not None
            else official_stats if official_stats.is_file() else checkpoint_stats
        )
        with stats_path.open() as stream:
            normalization = json.load(stream)
        normalization = normalization.get("norm_stats", normalization)
        state_stats = normalization["state"]
        action_stats = normalization["actions" if "actions" in normalization else "action"]
    elif norm_stats_path is not None:
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
    model = PI0Pytorch.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        fix_mistral_regex=True,
        local_files_only=dataset.embodiment == "libero",
    )
    transform_spec = None
    training_dataset = dataset.dataset
    collate_fn = dataset.collate_fn
    if dataset.embodiment == "libero":
        if model.config.action_horizon != 10 or model.config.action_dim < 7:
            raise ValueError("LIBERO requires action_horizon=10 and action_dim>=7")
        transform_spec = create_libero_transform_spec(
            tokenizer,
            {"state": state_stats, "actions": action_stats},
            model_action_dim=model.config.action_dim,
        )
        training_dataset = Pi05TransformedDataset(dataset.dataset, transform_spec)
        collate_fn = None
    elif dataset.embodiment == "so101":
        transform_spec = create_so101_transform_spec(
            tokenizer,
            {"state": state_stats, "actions": action_stats},
            model_action_dim=model.config.action_dim,
            discrete_state_input=model.config.discrete_state_input,
        )
        training_dataset = Pi05TransformedDataset(dataset.dataset, transform_spec)
        collate_fn = None
    elif dataset.embodiment != "robotwin":
        raise ValueError(f"unknown PI0.5 embodiment: {dataset.embodiment}")
    loss_fn = Pi05SFTLossFn(
        tokenizer,
        state_stats=state_stats,
        action_stats=action_stats,
        image_keys=dataset.image_keys,
        state_key=dataset.state_key,
        action_key=dataset.action_key,
        task_key=dataset.task_key,
        preprocess=load_callable(preprocess) if preprocess is not None else None,
        transform_spec=transform_spec,
    )
    return Pi05Components(
        model=model.to(device),
        loss_fn=loss_fn,
        dataset=training_dataset,
        collate_fn=collate_fn,
    )

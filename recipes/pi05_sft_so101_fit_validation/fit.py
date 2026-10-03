"""Recipe-only controls for the progressive SO101 fit experiments."""

from dataclasses import replace
from typing import Any, override

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.utils.data import Subset

from carrot.data.dataset_spec import SFTDatasetSpec
from carrot.data.so101 import build_dataset as build_so101_dataset
from carrot.models.pi05.loss_fn import Pi05SFTLossFn
from carrot.trainer.sft.config import SFTConfig
from carrot.trainer.sft.worker import SFTTrainWorker


def build_dataset(*, frame_index: int | None = None, repeat_count: int = 64,
                  **kwargs: Any) -> SFTDatasetSpec:
    """Select a source frame, preserving the source statistics and targets.

    Parameters
    ----------
    frame_index : int | None
        None keeps the complete source episode.
    repeat_count : int
        Virtual length for distributed single-frame batches.
    kwargs : Any
        Forwarded to the SO101 factory.

    Returns
    -------
    SFTDatasetSpec
    """
    spec = build_so101_dataset(**kwargs)
    if frame_index is None:
        return spec
    assert 0 <= frame_index < len(spec.dataset) and repeat_count >= 1
    return replace(spec, dataset=Subset(spec.dataset, [frame_index] * repeat_count))


class DropVision:
    """Remove image values and image attention masks after shared transforms."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return data | {
            "image": {key: np.zeros_like(value) for key, value in data["image"].items()},
            "image_mask": {key: False for key in data["image_mask"]},
        }


def fixed_noise(seed: int, horizon: int, width: int) -> np.ndarray:
    """Generate the identical BF16-representable latent for training and inference."""
    values = np.random.default_rng(seed).standard_normal((horizon, width)).astype(np.float32)
    return torch.from_numpy(values).to(torch.bfloat16).float().numpy()


class FitLoss(Pi05SFTLossFn):
    """Keep the production objective; optionally fix noise and report active axes."""

    def __init__(self, source: Pi05SFTLossFn, noise_seed: int | None) -> None:
        super().__init__(
            source.tokenizer, state_stats=source.state_stats, action_stats=source.action_stats,
            image_keys=source.image_keys, state_key=source.state_key,
            action_key=source.action_key, task_key=source.task_key,
            preprocess=source.preprocess, transform_spec=source.transform_spec,
        )
        self.noise_seed = noise_seed
        self.calls = 0

    @override
    def __call__(
        self, model: torch.nn.Module, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        observation, actions, valid = self.prepare_inputs(model, batch)
        if self.noise_seed is None:
            noise = torch.randn_like(actions)
        else:
            latent = fixed_noise(self.noise_seed, actions.shape[1], actions.shape[2])
            noise = torch.as_tensor(latent, device=actions.device, dtype=actions.dtype)
            noise = noise[None].expand_as(actions)
        gamma1 = torch.rand(actions.shape[0], device=actions.device).pow(1 / 1.5)
        gamma2 = torch.rand(actions.shape[0], device=actions.device)
        time = (gamma1 / (gamma1 + gamma2) * 0.999 + 0.001).to(actions.dtype)
        errors = model(observation, actions, noise=noise, time=time)
        per_step = errors.mean(dim=-1)
        loss = (per_step * valid).sum() / valid.sum()
        self.calls += 1
        if (self.calls == 1 or self.calls % 400 == 0) and (
            not dist.is_initialized() or dist.get_rank() == 0
        ):
            axes = ((errors.float() * valid[..., None]).sum(dim=(0, 1)) / valid.sum())
            print("FIT_ACTIVE_AXES", self.calls, axes[:6].detach().tolist(),
                  "PADDED_MEAN", axes[6:].mean().detach().item(), flush=True)
        return loss, {"loss": loss.detach(), "per_step_loss": per_step.detach()}


class FitTrainWorker(SFTTrainWorker):
    """Reuse the production worker with explicit recipe-local ablations."""

    def __init__(self, config: SFTConfig, vision: bool, noise_seed: int | None) -> None:
        super().__init__(config)
        self.vision = vision
        self.noise_seed = noise_seed

    @override
    def setup(self) -> None:
        super().setup()
        assert self.impl is not None
        # Horizon changes sequence length, not weight shapes; set before the first batch.
        horizon = self.config.dataset.factory_kwargs["action_horizon"]
        self.impl.model.register_to_config(action_horizon=horizon)
        loss = self.impl.loss_fn
        assert isinstance(loss, Pi05SFTLossFn) and loss.transform_spec is not None
        if not self.vision:
            spec = replace(loss.transform_spec, inputs=loss.transform_spec.inputs + (DropVision(),))
            loss.transform_spec = spec
            from_transforms = self.impl.dataloader.dataset.input_transform
            # The loader has not started its child processes yet.
            self.impl.dataloader.dataset.input_transform = replace(
                from_transforms, transforms=spec.inputs,
            )
        self.impl.loss_fn = FitLoss(loss, self.noise_seed)
        self.impl.checkpoint_artifact_writer = self.impl.loss_fn.save_artifacts

    @override
    def train(self) -> dict[str, float | int]:
        assert self.impl is not None
        weight = self.impl.model.action_out_proj.weight
        before = (weight.to_local() if isinstance(weight, DTensor) else weight).detach().clone()
        result = super().train()
        after = weight.to_local() if isinstance(weight, DTensor) else weight
        delta = (after.detach().float() - before.float()).abs()
        assert torch.isfinite(delta).all() and delta.max() > 0, "action head did not update"
        print("FIT_PARAMETER_UPDATE", self.rank, "dtype", after.dtype,
              "max", delta.max().item(), "mean", delta.mean().item(), flush=True)
        return result

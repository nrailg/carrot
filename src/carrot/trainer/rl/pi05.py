from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import default_collate

from carrot.models.pi05 import transforms
from carrot.models.pi05.model import PI0Observation, PI0Pytorch
from carrot.models.pi05.model.pi0_pytorch import make_att_2d_masks
from carrot.trainer.rl.algorithms import flow_transition, gaussian_logprob, supervised_flow_loss


@dataclass
class Prefix:
    state: torch.Tensor
    mask: torch.Tensor
    cache: Any
    pooled: torch.Tensor


class FlowPolicy:
    """Train the action expert through one scored Flow-SDE transition per chunk.

    Parameters
    ----------
    model : PI0Pytorch
        PI0.5 with horizon 10; VLM is frozen and trainable parameters retain FP32 masters.
    transform_spec : transforms.Pi05TransformSpec
    device : str
    num_steps : int
    noise_level : float
        Positive scale for training; evaluation uses the original ODE sampler.
    """

    def __init__(
        self,
        model: PI0Pytorch,
        transform_spec: transforms.Pi05TransformSpec,
        device: str = "cuda:0",
        num_steps: int = 5,
        noise_level: float = 0.5,
    ) -> None:
        assert model.config.pi05 and model.config.action_horizon == 10
        assert num_steps >= 2 and noise_level > 0
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.trainable = [
            model.paligemma_with_expert.gemma_expert,
            model.action_in_proj,
            model.action_out_proj,
            model.time_mlp_in,
            model.time_mlp_out,
        ]
        for module in self.trainable:
            module.float().requires_grad_(True)
        width = model.paligemma_with_expert.paligemma.language_model.config.hidden_size
        self.value_head = nn.Sequential(nn.Linear(width, 256), nn.Tanh(), nn.Linear(256, 1))
        self.value_head.to(self.device).float()
        self.inputs = transforms.compose(transform_spec.inputs)
        self.outputs = transforms.compose(transform_spec.outputs)
        self.num_steps, self.noise_level = num_steps, noise_level

    def encode(self, raw: list[dict]) -> tuple[PI0Observation, dict]:
        data = default_collate([self.inputs(dict(sample)) for sample in raw])
        return transforms.to_observation(data, device=self.device, dtype=torch.float32), data

    @torch.no_grad()
    def prefix(self, observation: PI0Observation) -> Prefix:
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            images, masks, tokens, token_masks, state = self.model._preprocess_observation(
                observation, train=False
            )
            embeddings, pad, att = self.model.embed_prefix(images, masks, tokens, token_masks)
            attention = self.model._prepare_attention_masks_4d(make_att_2d_masks(pad, att))
            backbone = self.model.paligemma_with_expert
            backbone.paligemma.language_model.config._attn_implementation = "eager"
            outputs, cache = backbone.forward(
                attention_mask=attention,
                position_ids=pad.cumsum(1) - 1,
                past_key_values=None,
                inputs_embeds=[embeddings, None],
                use_cache=True,
            )
        pooled = (outputs[0].float() * pad[..., None]).sum(1) / pad.sum(1, keepdim=True)
        return Prefix(state, pad, cache, pooled)

    def velocity(self, prefix: Prefix, latent: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            return self.model.denoise_step(
                prefix.state, prefix.mask, prefix.cache, latent, time
            ).float()

    @torch.no_grad()
    def sample(self, raw: list[dict], *, stochastic: bool = True) -> dict:
        """Return actions and replay data for the randomly selected stochastic transition.

        Parameters
        ----------
        raw : list[dict]
            Unnormalized observations; privileged simulator fields are excluded.
        stochastic : bool
            False uses ten ODE steps, matching the target evaluation protocol.

        Returns
        -------
        dict
            CPU tensors for replay and decoded action chunks of shape (B, 10, 7).
        """
        observation, _ = self.encode(raw)
        prefix = self.prefix(observation)
        batch = len(raw)
        latent = torch.randn(batch, 10, self.model.config.action_dim, device=self.device)
        steps = self.num_steps if stochastic else 10
        indices = torch.randint(steps, (batch,), device=self.device)
        before, after, logprob = (torch.zeros_like(latent) for _ in range(3))
        times = torch.linspace(1, 0, steps + 1, device=self.device)
        for step in range(steps):
            velocity = self.velocity(prefix, latent, times[step].expand(batch))
            next_latent = latent - (times[step] - times[step + 1]) * velocity
            if stochastic:
                index = torch.full_like(indices, step)
                mean, std = flow_transition(
                    latent, velocity, index, num_steps=steps, noise_level=self.noise_level
                )
                sampled = mean + std * torch.randn_like(mean)
                selected = (indices == step)[:, None, None]
                before = torch.where(selected, latent, before)
                after = torch.where(selected, sampled, after)
                logprob = torch.where(selected, gaussian_logprob(sampled, mean, std), logprob)
                next_latent = torch.where(selected, sampled, next_latent)
            latent = next_latent
        decoded = self.outputs({"state": observation.state, "actions": latent})["actions"]
        assert torch.isfinite(decoded).all()
        return {
            "actions": decoded.float().cpu().numpy(),
            "before": before.cpu(),
            "after": after.cpu(),
            "index": indices.cpu(),
            "logprob": logprob.cpu(),
            "value": self.value_head(prefix.pooled).squeeze(-1).cpu(),
        }

    def evaluate(self, raw: list[dict], replay: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Recompute the selected transition with gradients and identical preprocessing.

        Parameters
        ----------
        raw : list[dict]
        replay : dict
            Stored before/after latents and selected denoising indices, never final actions.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Per-element Gaussian log probabilities and scalar observation values.
        """
        observation, _ = self.encode(raw)
        prefix = self.prefix(observation)
        before, after, index = [replay[k].to(self.device) for k in ("before", "after", "index")]
        times = torch.linspace(1, 0, self.num_steps + 1, device=self.device)
        velocity = self.velocity(prefix, before, times[index])
        mean, std = flow_transition(
            before, velocity, index, num_steps=self.num_steps, noise_level=self.noise_level
        )
        return gaussian_logprob(after, mean, std), self.value_head(prefix.pooled).squeeze(-1)

    @torch.no_grad()
    def values(self, raw: list[dict]) -> torch.Tensor:
        observation, _ = self.encode(raw)
        return self.value_head(self.prefix(observation).pooled).squeeze(-1).cpu()

    def supervised_loss(self, raw: list[dict], lengths: torch.Tensor) -> torch.Tensor:
        observation, data = self.encode(raw)
        prefix = self.prefix(observation)
        actions = data["actions"].to(self.device, dtype=torch.float32)
        noise = torch.randn_like(actions)
        time = self.model.sample_time(len(raw), self.device)
        latent = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        prediction = self.velocity(prefix, latent, time)
        error = (prediction[:, :, :7] - (noise - actions)[:, :, :7]).square()
        return supervised_flow_loss(error, lengths)

    def parameters(self) -> list[nn.Parameter]:
        return [p for module in self.trainable for p in module.parameters()]

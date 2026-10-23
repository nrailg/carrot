# Flow-SDE coefficients adapted from RLinf (Apache-2.0), commit be1b0a4.
# Copyright 2026 The RLinf Authors. See recipes/pi05_multicolor_transfer/NOTICE.

import math

import torch


def flow_transition(
    latent: torch.Tensor,
    velocity: torch.Tensor,
    index: torch.Tensor,
    *,
    num_steps: int,
    noise_level: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the selected πRL Flow-SDE transition's Gaussian mean and standard deviation.

    Parameters
    ----------
    latent, velocity : torch.Tensor
        FP32 tensors with shape (batch, horizon, padded_action_dim).
    index : torch.Tensor
        Per-sample denoising index, in [0, num_steps).
    num_steps : int
    noise_level : float
        Strictly positive; deterministic ODE transitions are not scored.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Mean with the input shape and standard deviation with shape (batch, 1, 1).
    """
    assert num_steps >= 2 and noise_level > 0
    assert latent.shape == velocity.shape and latent.ndim == 3
    assert index.shape == latent.shape[:1] and index.dtype == torch.long
    assert torch.isfinite(latent).all() and torch.isfinite(velocity).all()
    assert ((index >= 0) & (index < num_steps)).all()
    times = torch.linspace(1, 0, num_steps + 1, device=latent.device, dtype=torch.float32)
    time = times[index, None, None]
    delta = times[index, None, None] - times[index + 1, None, None]
    denominator_time = torch.where(index[:, None, None] == 0, times[1], time)
    sigma = noise_level * torch.sqrt(time / (1 - denominator_time))
    x0 = latent.float() - time * velocity.float()
    x1 = latent.float() + (1 - time) * velocity.float()
    mean = x0 * (1 - time + delta) + x1 * (time - delta - sigma.square() * delta / (2 * time))
    return mean, delta.sqrt() * sigma


def gaussian_logprob(sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    assert (std > 0).all() and torch.isfinite(std).all()
    result = -std.log() - 0.5 * math.log(2 * math.pi) - 0.5 * ((sample - mean) / std).square()
    assert torch.isfinite(result).all(), "non-finite Flow-SDE log probabilities"
    return result


def chunk_logprob(
    logprob: torch.Tensor, lengths: torch.Tensor, action_dim: int = 7
) -> torch.Tensor:
    assert logprob.ndim == 3 and lengths.shape == logprob.shape[:1]
    assert lengths.dtype == torch.long
    assert ((lengths > 0) & (lengths <= logprob.shape[1])).all()
    assert 0 < action_dim <= logprob.shape[-1]
    mask = torch.arange(logprob.shape[1], device=logprob.device)[None] < lengths[:, None]
    return (logprob[:, :, :action_dim] * mask[:, :, None]).sum(dim=(1, 2))


def duration_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    lengths: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute SMDP GAE without leaking advantages across explicit episode resets.

    Parameters
    ----------
    rewards : torch.Tensor
        Shape (time, env), already discounted within each executed chunk.
    values, next_values : torch.Tensor
        Values before the chunk and at its final observation, before any reset.
    lengths : torch.Tensor
        Actual control steps in each chunk; all other inputs have the same shape.
    terminated, truncated : torch.Tensor
        Both stop the trace; only terminated disables bootstrapping.
    gamma, gae_lambda : float
        Per-control-step factors, exponentiated by each chunk duration.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Advantages and critic targets with shape (time, env).
    """
    assert rewards.ndim == 2 and 0 < gamma <= 1 and 0 <= gae_lambda <= 1
    assert rewards.numel() > 0 and lengths.dtype == torch.long
    assert terminated.dtype == truncated.dtype == torch.bool
    assert all(torch.isfinite(x).all() for x in (rewards, values, next_values))
    assert all(
        x.shape == rewards.shape for x in (values, next_values, lengths, terminated, truncated)
    )
    assert (lengths > 0).all()
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        discount = gamma ** lengths[t]
        delta = rewards[t] + discount * next_values[t] * ~terminated[t] - values[t]
        carry = delta + (gamma * gae_lambda) ** lengths[t] * ~(terminated[t] | truncated[t]) * carry
        advantages[t] = carry
    return advantages, advantages + values


def ppo_loss(
    logprob: torch.Tensor,
    old_logprob: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    clip: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    assert (
        logprob.shape
        == old_logprob.shape
        == advantages.shape
        == values.shape
        == old_values.shape
        == returns.shape
    )
    assert 0 < clip < 1
    log_ratio = logprob - old_logprob
    ratio = log_ratio.exp()
    assert torch.isfinite(ratio).all(), "PPO ratio overflow; stop rather than silently clamp"
    policy_loss = -torch.minimum(
        ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages
    ).mean()
    clipped_value = old_values + (values - old_values).clamp(-clip, clip)
    value_loss = (
        0.5 * torch.maximum((values - returns).square(), (clipped_value - returns).square()).mean()
    )
    return policy_loss + value_loss, {
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "approx_kl": ((ratio - 1) - log_ratio).mean().detach(),
        "clip_fraction": ((ratio - 1).abs() > clip).float().mean().detach(),
    }

"""Small PPO update used only to verify the simulator-to-learner contract."""

import torch
from torch import nn
from torch.distributions import Normal

from carrot_sim.libero import LiberoVectorEnv, Observation


def generalized_advantage(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:
    # 超时使用最终状态 bootstrap，但不能把下一回合的优势递推回来。
    deltas = rewards + gamma * (~terminated) * next_values - values
    advantage = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for step in reversed(range(rewards.shape[0])):
        carry = deltas[step] + gamma * lam * (~(terminated[step] | truncated[step])) * carry
        advantage[step] = carry
    return advantage


def _features(observation: Observation) -> torch.Tensor:
    policy = observation["policy"]
    return torch.cat(
        [
            policy["state"],
            policy["image"].float().mean(dim=(1, 2)) / 255,
            policy["wrist_image"].float().mean(dim=(1, 2)) / 255,
        ],
        dim=-1,
    )


def _log_prob(distribution: Normal, latent: torch.Tensor) -> torch.Tensor:
    return (distribution.log_prob(latent) - torch.log(1 - latent.tanh().square() + 1e-6)).sum(-1)


def run_ppo_check(env: LiberoVectorEnv, steps: int = 32) -> dict[str, float | int]:
    # 使用图像和机器人状态驱动 actor，特权物体状态仅提供给 critic。
    actor = nn.Sequential(nn.Linear(14, 32), nn.Tanh(), nn.Linear(32, 7)).to(env.device)
    critic = nn.Sequential(nn.Linear(52, 32), nn.Tanh(), nn.Linear(32, 1)).to(env.device)
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=3e-4)
    before_actor = torch.cat([p.detach().flatten().clone() for p in actor.parameters()])
    before_critic = torch.cat([p.detach().flatten().clone() for p in critic.parameters()])
    records = []
    observation, _ = env.reset()

    # 保存每个 transition 的终止前值估计；不使用 autoreset 后的新回合状态。
    with torch.no_grad():
        for _ in range(steps):
            features = _features(observation)
            critic_input = observation["critic"]
            distribution = Normal(actor(features), 0.25)
            latent = distribution.sample()
            value = critic(critic_input).squeeze(-1)
            next_obs, reward, terminated, truncated, info = env.step(latent.tanh())
            next_input = next_obs["critic"].clone()
            mask = info["_final_observation"]
            if mask.any():
                next_input[mask] = info["final_observation"]["critic"][mask]
            next_value = critic(next_input).squeeze(-1)
            records.append(
                (
                    features,
                    critic_input,
                    latent,
                    _log_prob(distribution, latent),
                    value,
                    reward,
                    terminated,
                    truncated,
                    next_value,
                )
            )
            observation = next_obs
    (
        features,
        critic_inputs,
        latents,
        old_log_prob,
        values,
        rewards,
        terminated,
        truncated,
        next_values,
    ) = (torch.stack(items) for items in zip(*records, strict=True))
    advantages = generalized_advantage(rewards, values, next_values, terminated, truncated)
    returns = advantages + values
    normalized = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)

    # 两轮更新会实际触发 PPO ratio，而不是只验证一次普通反向传播。
    for _ in range(2):
        distribution = Normal(actor(features), 0.25)
        ratio = (_log_prob(distribution, latents) - old_log_prob).exp()
        policy_loss = -torch.minimum(ratio * normalized, ratio.clamp(0.8, 1.2) * normalized).mean()
        value_loss = (critic(critic_inputs).squeeze(-1) - returns).square().mean()
        loss = policy_loss + 0.5 * value_loss
        assert torch.isfinite(loss), "PPO loss contains NaN/Inf"
        optimizer.zero_grad()
        loss.backward()
        parameters = [*actor.parameters(), *critic.parameters()]
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()

    # actor 和 critic 都必须发生有限更新；稀疏 reward 为零不等于任务已经学会。
    after_actor = torch.cat([p.detach().flatten() for p in actor.parameters()])
    after_critic = torch.cat([p.detach().flatten() for p in critic.parameters()])
    actor_delta = (after_actor - before_actor).abs().max().item()
    critic_delta = (after_critic - before_critic).abs().max().item()
    assert torch.isfinite(after_actor).all() and torch.isfinite(after_critic).all()
    assert actor_delta > 0 and critic_delta > 0
    return {
        "transitions": steps * env.num_envs,
        "terminated": terminated.sum().item(),
        "truncated": truncated.sum().item(),
        "reward_sum": rewards.sum().item(),
        "loss": loss.item(),
        "actor_max_update": actor_delta,
        "critic_max_update": critic_delta,
    }

import torch


def ppo_smoke(env, steps: int) -> dict:
    # 采样真实转移；actor 只读机器人状态，critic 读特权状态。
    actor = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.Tanh(), torch.nn.Linear(32, 7)).to(
        env.device
    )
    critic = torch.nn.Sequential(
        torch.nn.Linear(env.single_observation_space["critic"].shape[0], 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 1),
    ).to(env.device)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    initial_actor = torch.cat([p.detach().flatten().clone() for p in actor.parameters()])
    initial_critic = torch.cat([p.detach().flatten().clone() for p in critic.parameters()])
    data = []
    obs = env.reset(seed=42)
    for _ in range(steps):
        with torch.no_grad():
            dist = torch.distributions.Normal(actor(obs["state"]), 0.25)
            latent = dist.sample()
            old_logp = dist.log_prob(latent).sum(-1)
            value = critic(obs["critic"]).squeeze(-1)
            new, reward, term, trunc, info = env.step(latent.tanh())
            boot = new["critic"].clone()
            if info["final_observation"] is not None:
                mask = info["final_mask"]
                boot[mask] = info["final_observation"]["critic"][mask]
            next_value = critic(boot).squeeze(-1) * info["bootstrap_mask"]
        data.append((obs, latent, old_logp, value, reward, term, trunc, next_value))
        obs = new

    # GAE 在 episode 边界停止递推，纯超时仍用 terminal critic bootstrap。
    advantages = []
    gae = torch.zeros(env.num_envs, device=env.device)
    for _, _, _, value, reward, term, trunc, next_value in reversed(data):
        delta = reward + 0.99 * next_value - value
        gae = delta + 0.99 * 0.95 * (~(term | trunc)) * gae
        advantages.append(gae.clone())
    advantage = torch.stack(list(reversed(advantages))).flatten()
    old_value = torch.stack([row[3] for row in data]).flatten()
    returns = advantage + old_value
    advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-6)
    states = torch.cat([row[0]["state"] for row in data])
    privileged = torch.cat([row[0]["critic"] for row in data])
    latents = torch.cat([row[1] for row in data])
    old_logp = torch.cat([row[2] for row in data])

    # 两次优化实际检查有限梯度和 actor/critic 参数变化。
    for _ in range(2):
        dist = torch.distributions.Normal(actor(states), 0.25)
        ratio = (dist.log_prob(latents).sum(-1) - old_logp).exp()
        actor_loss = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage).mean()
        loss = actor_loss + 0.5 * (critic(privileged).squeeze(-1) - returns).square().mean()
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        params = list(actor.parameters()) + list(critic.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
    actor_update = (
        (torch.cat([p.detach().flatten() for p in actor.parameters()]) - initial_actor)
        .abs()
        .max()
        .item()
    )
    critic_update = (
        (torch.cat([p.detach().flatten() for p in critic.parameters()]) - initial_critic)
        .abs()
        .max()
        .item()
    )
    assert actor_update > 0 and critic_update > 0
    truncations = sum(row[6].sum().item() for row in data)
    assert truncations > 0
    return {
        "transitions": steps * env.num_envs,
        "truncations": truncations,
        "loss": loss.item(),
        "actor_update": actor_update,
        "critic_update": critic_update,
    }

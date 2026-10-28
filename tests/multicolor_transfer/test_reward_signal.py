import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from carrot.trainer.rl.algorithms import distribution_statistics, duration_gae
from carrot.trainer.rl.config import load_config
from carrot.trainer.rl.data import Demonstrations
from carrot.trainer.rl.learner import LearnerWorker


def check_credit_scale(source: Path, output: Path) -> None:
    # 只复用已有reward/value，逐控制步比较折扣尺度，不重新推理或改正式超参数。
    signals = json.loads((source / "reward-events.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for episode, signal in enumerate(signals):
        count = signal["controls"]
        rewards = torch.zeros(count, 1)
        for offset, reward in signal["events"]:
            rewards[offset] = reward
        values = torch.from_numpy(np.load(source / f"episode-{episode}-values.npy"))[:, None]
        assert values.shape == rewards.shape
        following = torch.cat((values[1:], torch.zeros(1, 1)))
        lengths = torch.ones(count, 1, dtype=torch.long)
        terminated = torch.zeros(count, 1, dtype=torch.bool)
        terminated[-1] = True
        truncated = torch.zeros_like(terminated)
        modes = {}
        # 最后一个对照等价完整episode的未折扣回报，独立核对首动作信号等于总奖励。
        for name, gamma, lam in (
            ("current", 0.99, 0.95), ("lambda_one", 0.99, 1.0), ("mc", 1.0, 1.0)
        ):
            reward_adv, _ = duration_gae(
                rewards, torch.zeros_like(values), torch.zeros_like(values), lengths,
                terminated, truncated, gamma=gamma, gae_lambda=lam,
            )
            actual_adv, _ = duration_gae(
                rewards, values, following, lengths, terminated, truncated,
                gamma=gamma, gae_lambda=lam,
            )
            modes[name] = {
                "gamma": gamma,
                "lambda": lam,
                "initial_reward_advantage": float(reward_adv[0]),
                "first100_reward_mean": float(reward_adv[:100].mean()),
                "first100_reward_near_zero_fraction": float(
                    (reward_adv[:100].abs() <= 1e-8).float().mean()
                ),
                "first100_actual_advantage": distribution_statistics(actual_adv[:100]),
            }
            if name == "mc":
                torch.testing.assert_close(reward_adv[0, 0], rewards.sum())
        results.append({"seed": signal["seed"], "controls": count, "modes": modes})
    (output / "result.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps([
        {"seed": r["seed"], "initial_reward_advantage": {
            k: v["initial_reward_advantage"] for k, v in r["modes"].items()
        }} for r in results
    ]), flush=True)


def check_signal(config_path: str, checkpoint: str, signals: list[dict], output: Path) -> None:
    # 完整teacher轨迹的真实SFT critic与奖励分别贡献多少优势；这不是旧RL在线优势的复原。
    config = replace(load_config(config_path), learner_node_ip="127.0.0.1")
    assert config.learner_demonstrations is not None
    dataset = Demonstrations(config.learner_demonstrations)
    output.mkdir(parents=True, exist_ok=False)
    (output / "reward-events.json").write_text(json.dumps(signals) + "\n")
    os.environ.update(LOCAL_RANK="0", MASTER_ADDR="127.0.0.1")
    learner = LearnerWorker(config)
    learner.setup()
    learner.resume(checkpoint)
    policy = learner.policy
    assert policy is not None
    results, base = [], 0
    for episode, source in enumerate(signals):
        # 复用HZ原始观测，仅回放结果中的小型reward元数据来自Pro，不复制teacher图像。
        data = dataset.episodes[episode]
        count = len(data["actions"])
        episode_path = Path(config.learner_demonstrations) / f"episode-{episode:04d}"
        stored = json.loads((episode_path / "result.json").read_text())
        assert stored["seed"] == source["seed"]
        assert source["controls"] == count and source["success"]
        rewards = torch.zeros(count)
        for offset, reward in source["events"]:
            assert 0 <= offset < count and rewards[offset] == 0
            rewards[offset] = reward
        values = torch.cat(
            [
                policy.values(
                    [dataset[base + i][0] for i in range(start, min(start + 8, count))]
                )
                for start in range(0, count, 8)
            ]
        )
        np.save(output / f"episode-{episode}-values.npy", values.numpy())
        base += count
        # 按phase边界和最多5控制步展开；记录分段长度，不能假装复原旧32环境的切分时机。
        starts, lengths, offset = [], [], 0
        while offset < count:
            length = 1
            while (
                length < config.execute_steps
                and offset + length < count
                and data["phase"][offset + length] == data["phase"][offset]
            ):
                length += 1
            starts.append(offset)
            lengths.append(length)
            offset += length
        chunk_rewards = torch.tensor(
            [
                sum(config.gamma**i * rewards[start + i] for i in range(length))
                for start, length in zip(starts, lengths, strict=True)
            ]
        )[:, None]
        before = values[starts, None]
        after = torch.cat((values[starts[1:]], torch.zeros(1)))[:, None]
        duration = torch.tensor(lengths)[:, None]
        terminated = torch.zeros_like(chunk_rewards, dtype=torch.bool)
        terminated[-1] = True
        truncated = torch.zeros_like(terminated)
        advantages = {}
        for name, reward, value, following in (
            ("actual_critic", chunk_rewards, before, after),
            ("critic_only", torch.zeros_like(chunk_rewards), before, after),
            ("reward_only", chunk_rewards, torch.zeros_like(before), torch.zeros_like(after)),
        ):
            advantages[name], _ = duration_gae(
                reward,
                value,
                following,
                duration,
                terminated,
                truncated,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )
        # GAE线性分解必须成立；正负号本身不能证明策略获得了任务奖励反馈。
        torch.testing.assert_close(
            advantages["actual_critic"],
            advantages["critic_only"] + advantages["reward_only"],
            atol=1e-6,
            rtol=1e-5,
        )
        raw = advantages["actual_critic"]
        normalized = (raw - raw.mean()) / (raw.std(unbiased=False) + 1e-8)
        first = np.asarray(starts) < 100
        result = {
            "seed": source["seed"],
            "controls": count,
            "chunks": len(starts),
            "reward_sum": float(rewards.sum()),
            "nonzero_reward_controls": int(torch.count_nonzero(rewards)),
            "reward": distribution_statistics(rewards),
            "value": distribution_statistics(values),
            "advantages": {k: distribution_statistics(v) for k, v in advantages.items()},
            "normalized": distribution_statistics(normalized),
            "first_100_controls": {
                k: distribution_statistics(v[first]) for k, v in advantages.items()
            },
            "initial_reward_advantage": float(advantages["reward_only"][0]),
        }
        results.append(result)
        (output / "result.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    (output / "completed.json").write_text(
        json.dumps({"episodes": len(results), "source": "teacher_offpolicy_diagnostic"}) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--signals-json")
    parser.add_argument("--credit-input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.credit_input is not None:
        assert args.config is args.checkpoint is args.signals_json is None
        check_credit_scale(args.credit_input, args.output)
    else:
        assert args.config and args.checkpoint and args.signals_json
        check_signal(args.config, args.checkpoint, json.loads(args.signals_json), args.output)

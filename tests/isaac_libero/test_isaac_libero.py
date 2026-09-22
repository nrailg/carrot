import argparse
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from isaac_libero.rl_check import generalized_advantage, run_ppo_check
from PIL import Image, ImageStat

from carrot_sim.libero import IsaacLiberoConfig, LiberoVectorEnv, clone_observation


class FakeBackend:
    """CPU transition fixture; never used as evidence of working Isaac physics."""

    num_envs = 3
    device = "cpu"
    single_observation_space = gym.spaces.Dict({"critic": gym.spaces.Box(-np.inf, np.inf, (52,))})
    observation_space = gym.vector.utils.batch_space(single_observation_space, num_envs)

    def __init__(self):
        self.state = torch.zeros(self.num_envs, 8)
        self.age = torch.zeros(self.num_envs, dtype=torch.long)
        self.actions = None
        self.close_calls = 0

    def observe(self):
        return {
            "policy": {
                "state": self.state,
                "image": torch.full((self.num_envs, 2, 2, 3), 32, dtype=torch.uint8),
                "wrist_image": torch.full((self.num_envs, 2, 2, 3), 64, dtype=torch.uint8),
            },
            "critic": self.state[:, :1].expand(-1, 52),
        }

    def reset(self, *, env_ids):
        self.state[env_ids] = 0
        self.age[env_ids] = 0
        return self.observe(), {}

    def step(self, action):
        self.actions = action.clone()
        self.state += 1 + action[:, :1] * 0.1
        self.age += 1
        terminated = torch.tensor([True, False, True]) & (self.age >= 2)
        truncated = torch.tensor([False, True, True]) & (self.age >= 2)
        done = terminated | truncated
        final = clone_observation(self.observe()) if done.any() else None
        self.state[done] = -10
        self.age[done] = 0
        return (
            self.observe(),
            torch.zeros(3),
            terminated,
            truncated,
            {
                "success": terminated.clone(),
                "final_observation": final,
            },
        )

    def close(self):
        self.close_calls += 1


def test_action_mapping_and_observation_ownership():
    # 验证夹爪方向与输入不变性，并防止 rollout 缓存被后续仿真原地覆盖。
    backend = FakeBackend()
    env = LiberoVectorEnv(backend)
    initial, _ = env.reset()
    action = torch.zeros(3, 7)
    action[:, :6] = torch.linspace(-0.5, 0.5, 6)
    action[:, 6] = torch.tensor([-1.0, 0.0, 1.0])
    saved = action.clone()

    # 经过生产适配器推进两步，backend 会原地改变状态并自动重置。
    first, *_ = env.step(action)
    env.step(action)

    # 只有夹爪符号映射给原生 IK，历史状态和调用方动作仍保留原值。
    torch.testing.assert_close(backend.actions[:, 6], torch.tensor([1.0, -1.0, -1.0]))
    torch.testing.assert_close(backend.actions[:, :6], action[:, :6])
    torch.testing.assert_close(action, saved)
    assert (initial["policy"]["state"] == 0).all()
    torch.testing.assert_close(first["policy"]["state"], (1 + saved[:, :1] * 0.1).expand(-1, 8))


def test_success_timeout_and_final_observation():
    # 三个槽分别成功、超时、同时成功与超时，保护 reward 和 bootstrap 的边界。
    env = LiberoVectorEnv(FakeBackend())
    env.reset()
    action = torch.zeros(3, 7)
    env.step(action)

    # 完成回合后保存最终状态，再推进一步以检测返回张量是否持有后端视图。
    obs, reward, terminated, truncated, info = env.step(action)
    env.step(action)

    # 同时成功和超时按真正终止处理；最终状态必须来自重置前而非 -10 的新状态。
    torch.testing.assert_close(reward, torch.tensor([1.0, 0.0, 1.0]))
    assert terminated.tolist() == [True, False, True]
    assert truncated.tolist() == [False, True, True]
    assert info["time_outs"].tolist() == [False, True, False]
    assert info["_final_observation"].all() and info["_episode"].all()
    assert (obs["policy"]["state"] == -10).all()
    assert (info["final_observation"]["policy"]["state"] == 2).all()
    assert info["episode"]["l"].tolist() == [2, 2, 2]
    torch.testing.assert_close(info["episode"]["r"], reward)


def test_partial_reset_and_lifecycle():
    # 部分重置只能影响选中的槽；首次不完整 reset 和关闭后的调用必须报错。
    backend = FakeBackend()
    env = LiberoVectorEnv(backend)
    with pytest.raises(RuntimeError, match="first reset"):
        env.reset(torch.tensor([1]))
    env.reset()
    env.step(torch.zeros(3, 7))

    # 重置中间槽，其他槽的状态与步数保留。
    obs, _ = env.reset(torch.tensor([1]))
    assert obs["policy"]["state"][:, 0].tolist() == [1, 0, 1]
    _, _, _, _, info = env.step(torch.zeros(3, 7))
    assert info["episode"]["l"].tolist() == [2, 1, 2]

    # close 可重复调用且只释放一次后端。
    env.close()
    env.close()
    assert backend.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()


def test_previous_success_is_not_rewarded_again(monkeypatch):
    # Isaac 的 term 历史会保留旧成功；其他槽结束时不能再次奖励正在运行的槽。
    backend = FakeBackend()
    env = LiberoVectorEnv(backend)
    env.reset()
    env.step(torch.zeros(3, 7))
    env.reset(torch.tensor([1]))
    env.step(torch.zeros(3, 7))
    original_step = backend.step

    # 模拟槽 0/2 上一回合已成功、槽 1 本步超时，term 历史尚未清除。
    def step_with_history(action):
        obs, reward, terminated, truncated, info = original_step(action)
        info["success"] = torch.tensor([True, False, True])
        return obs, reward, terminated, truncated, info

    monkeypatch.setattr(backend, "step", step_with_history)
    _, reward, terminated, truncated, info = env.step(torch.zeros(3, 7))

    # 本步没有新成功，只有槽 1 超时，因此 reward 和 success 都必须为零。
    assert not terminated.any() and truncated.tolist() == [False, True, False]
    assert not info["success"].any() and not reward.any()


@pytest.mark.parametrize(
    "ids",
    [
        torch.tensor([0, 0]),
        torch.tensor([-1]),
        torch.tensor([3]),
        torch.tensor([1.0]),
        torch.tensor([], dtype=torch.long),
    ],
)
def test_invalid_reset_ids(ids):
    # 拒绝重复、越界、错误 dtype 与空列表，避免静默重置错误槽。
    env = LiberoVectorEnv(FakeBackend())
    env.reset()

    # 输入校验必须在调用后端之前触发。
    with pytest.raises(ValueError, match="env_ids"):
        env.reset(ids)


@pytest.mark.parametrize(
    "action",
    [
        torch.zeros(3, 8),
        torch.zeros(3, 7, dtype=torch.float64),
        torch.full((3, 7), float("nan")),
        torch.full((3, 7), 1.1),
    ],
)
def test_invalid_actions(action):
    # 动作 shape、dtype、有限性和归一化范围属于显式接口，不能静默截断或广播。
    env = LiberoVectorEnv(FakeBackend())
    env.reset()

    # 非法输入不会送入模拟器。
    with pytest.raises(ValueError, match="action"):
        env.step(action)


def test_gae_timeout_bootstrap_and_episode_boundary():
    # 用可手算轨迹检验超时 bootstrap、成功不 bootstrap、回合间不串接优势。
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.zeros_like(rewards)
    next_values = torch.full_like(rewards, 5.0)
    terminated = torch.tensor([[False], [True], [False]])
    truncated = torch.tensor([[True], [True], [False]])

    # 第一项超时为 1+0.9*5，第二项真正终止为 2，第三项截断 rollout 仍 bootstrap。
    actual = generalized_advantage(
        rewards, values, next_values, terminated, truncated, gamma=0.9, lam=1.0
    )
    torch.testing.assert_close(actual, torch.tensor([[5.5], [2.0], [7.5]]))


def test_cpu_ppo_contract():
    # CPU fixture 仅验证 rollout/GAE/PPO 接口可串通，不能替代 Isaac GPU 验收。
    torch.manual_seed(7)
    env = LiberoVectorEnv(FakeBackend())

    # 真实执行 autograd 与优化器，并分别确认 actor/critic 发生有限更新。
    result = run_ppo_check(env, steps=8)
    assert result["transitions"] == 24
    assert result["actor_max_update"] > 0 and result["critic_max_update"] > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_envs": 0},
        {"image_size": -1},
        {"max_episode_steps": 1.5},
        {"rotation_scale": float("inf")},
    ],
)
def test_invalid_configuration(kwargs):
    # 在启动重型 Isaac runtime 前拒绝无法定义的批量尺寸、步数和动作单位。
    with pytest.raises(ValueError):
        IsaacLiberoConfig(**kwargs)


def validate_result(output: Path) -> None:
    # Kit 的退出码可能掩盖 Python 失败，必须独立验证本次生成的完成标记和产物。
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "PASS" and result["task"] == LiberoVectorEnv.task_name
    assert result["partial_reset"] == result["terminal_observation"] == "PASS"
    assert result["success_state_fixture"] == "PASS"
    assert result["num_envs"] == 4 and result["steps"] == 32
    ppo = result["ppo"]
    assert ppo["transitions"] == 128 and ppo["truncated"] > 0
    assert math.isfinite(ppo["loss"])
    for name in ("actor_max_update", "critic_max_update"):
        assert math.isfinite(ppo[name]) and ppo[name] > 0

    # 检查两个视角均落盘为有内容的 RGB；任务物体是否入画仍需人工复核。
    for name in ("image", "wrist_image"):
        with Image.open(output / f"{name}.png") as image:
            assert image.size == (256, 256) and image.mode == "RGB"
            assert max(ImageStat.Stat(image).stddev) > 1
    print("PASS: real Isaac LIBERO rollout, reset semantics, and PPO update")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    validate_result(parser.parse_args().result)

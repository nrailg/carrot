import argparse
import json
import math
from pathlib import Path

import pytest
import torch

from carrot_sim.arena_libero.config import ArenaLiberoConfig
from carrot_sim.arena_libero.predicates import bowl_on_plate
from carrot_sim.arena_libero.vector_env import ArenaLiberoEnv


class FakeBackend:
    num_envs = 2
    device = "cpu"

    def __init__(self):
        self.observation = {
            "policy": {
                "state": torch.zeros(2, 8),
                "image": torch.zeros(2, 4, 4, 3, dtype=torch.uint8),
            },
            "critic": torch.zeros(2, 52),
        }
        self.action = None

    def reset(self, *, seed=None, env_ids=None):
        self.ids = env_ids
        return self.observation, {}

    def step(self, action):
        self.action = action
        return (
            self.observation,
            torch.tensor([1.0, 0.0]),
            torch.tensor([True, False]),
            torch.tensor([False, True]),
            {"success": torch.tensor([True, False]), "final_observation": self.observation},
        )

    def close(self):
        pass


def test_owned_observations_and_bootstrap():
    # rollout 必须持有独立快照；成功不 bootstrap，纯超时必须 bootstrap。
    backend = FakeBackend()
    env = ArenaLiberoEnv(backend)
    action = torch.zeros(2, 7)
    action[0, 6] = -1

    # 同时覆盖夹爪符号映射与成功、超时两个终止分支。
    obs, reward, terminated, truncated, info = env.step(action)
    backend.observation["policy"]["state"].fill_(3)

    # 后续环境写入不得污染普通观测或 terminal 观测，也不能改动输入动作。
    assert obs["state"].count_nonzero() == 0
    assert info["final_observation"]["state"].count_nonzero() == 0
    assert info["bootstrap_mask"].tolist() == [False, True]
    assert info["final_mask"].tolist() == [True, True]
    assert backend.action[:, 6].tolist() == [1.0, -1.0]
    assert action[:, 6].tolist() == [-1.0, 0.0]
    assert terminated.tolist() == [True, False]
    assert truncated.tolist() == [False, True]
    assert reward.tolist() == [1.0, 0.0]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1.01, -1.01])
def test_invalid_action(bad):
    # 拒绝无效输入，避免 NaN 或越界动作进入 PhysX。
    env = ArenaLiberoEnv(FakeBackend())
    action = torch.zeros(2, 7)
    action[0, 0] = bad

    # 校验必须在调用 backend 前失败。
    with pytest.raises(ValueError):
        env.step(action)
    assert env.backend.action is None


def test_partial_reset_validation():
    # 部分 reset 保留明确的 slot 身份，拒绝重复或越界索引。
    backend = FakeBackend()
    env = ArenaLiberoEnv(backend)
    selected = torch.tensor([1])

    # 合法子集按原样传递，不扩大成全量 reset。
    env.reset(selected)
    assert torch.equal(backend.ids, selected)

    # 重复、越界和错误 dtype 都不能进入模拟器。
    for invalid in (torch.tensor([1, 1]), torch.tensor([2]), torch.tensor([0.0])):
        with pytest.raises(ValueError):
            env.reset(invalid)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_environment_count(value):
    # 非正整数并行度必须在启动 Kit 前被发现。
    with pytest.raises(ValueError):
        ArenaLiberoConfig(asset_root=Path("/unused"), num_envs=value)


def test_placement_requires_support_release_and_rest():
    # 水平接近不能算成功：悬空、移动、夹紧和盘外均应被拒绝。
    position = torch.tensor([[0.0, 0.0, 0.04]]).repeat(6, 1)
    force = torch.tensor([[0.0, 0.0, 0.5]]).repeat(6, 1)
    velocity = torch.zeros(6, 3)
    fingers = torch.full((6, 2), 0.04)

    # 每个反例只破坏一个必要条件，防止宽松判定掩盖错误。
    force[1] = 0
    velocity[2, 0] = 0.2
    fingers[3] = 0
    position[4, 0] = 0.2
    position[5, 2] = -0.04

    # 只有位于盘上且有支撑、静止并松开的样本成立。
    assert bowl_on_plate(position, force, velocity, fingers).tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]


def validate_result(output: Path) -> None:
    # 由独立 Python 进程复核完整结果，防止 Kit 快速退出掩盖验收失败。
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "PASS"
    assert result["versions"]["python"].startswith("3.12.")
    assert result["versions"]["isaacsim"] == "6.0.1.0"
    assert result["versions"]["isaaclab"] == "3.0.0b2.post1"
    assert result["versions"]["isaaclab-arena"] == "0.3.0"
    assert not result["lw_imported"]
    assert result["reset_wrist_mae"] < 20.0
    assert result["repeat_reset_wrist_mae"] < 20.0
    assert result["tcp_motion_min"] > 0.01
    assert result["finger_motion_min"] > 0.01

    # PPO 参数更新与终止语义必须同时满足，图片产物需完整落盘。
    for name in ("success_fixture", "partial_reset", "terminal_observation"):
        assert result[name] == "PASS"
    ppo = result["ppo"]
    assert ppo["transitions"] >= 128 and ppo["truncations"] > 0
    assert math.isfinite(ppo["loss"])
    for name in ("actor_update", "critic_update"):
        assert math.isfinite(ppo[name]) and ppo[name] > 0
    for name in ("image.png", "wrist_image.png"):
        assert (output / name).stat().st_size > 1024
    print("Independent result validation PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    validate_result(parser.parse_args().output)

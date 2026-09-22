import argparse
import json
from pathlib import Path

import pytest
import torch

from carrot_sim.arena_libero.predicates import (
    inside_drawer_and_closed,
    released_and_stable,
    stove_is_on,
    supported_on,
)
from carrot_sim.arena_libero.tasks import get_task, list_tasks


def test_drawer_requires_containment_and_closed_per_environment():
    # 合取必须逐环境计算；只关抽屉或只把碗放入均不能成功。
    position = torch.zeros(4, 3)
    position[2, 0] = 0.09
    position[3, 2] = 0.05
    joints = torch.tensor([0.0, 0.2, 0.0, 0.0])

    # 碗中心虽在范围内，但碗边缘越界的样本也应失败。
    result = inside_drawer_and_closed(
        position,
        torch.tensor([0.03, 0.03, 0.02]),
        torch.tensor([0.1, 0.1, 0.06]),
        joints,
        0.0,
        0.01,
    )
    assert result.tolist() == [True, False, False, False]


def test_stove_requires_on_and_supported_per_environment():
    # 防止将 env 0 的成功广播到整个 batch，也拒绝关火和悬空的壶。
    angle = torch.tensor([0.7, 0.0, 0.7, 0.7, 6.2])
    position = torch.tensor([[0.0, 0.0, 0.05]]).repeat(5, 1)
    position[2, 0] = 0.3
    force = torch.tensor([[0.0, 0.0, 0.5]]).repeat(5, 1)
    force[3] = 0

    # 分别破坏旋钮、位置或支撑条件，只有第一行应通过。
    result = stove_is_on(angle) & supported_on(position, torch.tensor([0.1, 0.1, 0.2]), force)
    assert result.tolist() == [True, False, False, False, False]


def test_release_uses_tcp_distance_and_object_speed():
    # 夹爪张开但仍靠近物体，或物体仍在运动，都不能算任务完成。
    obj = torch.zeros(3, 3)
    tcp = torch.tensor([[0.2, 0, 0], [0.01, 0, 0], [0.2, 0, 0]])
    velocity = torch.zeros(3, 3)
    velocity[2, 0] = 0.1

    # 验证距离与速度共同约束，不依赖夹爪命令符号。
    assert released_and_stable(obj, tcp, velocity).tolist() == [True, False, False]


def test_catalog_has_five_distinct_task_files():
    # 截图的五个任务应可枚举并一一对应文件，不能用同一场景的别名冒充。
    tasks = list_tasks()
    assert len(tasks) == 5
    assert len(list_tasks("libero_spatial")) == 3
    assert len(list_tasks("libero_10")) == 2

    # 注册名与文件路径保持一致，未知任务必须明确失败。
    root = Path(__file__).parents[2] / "src/carrot_sim/arena_libero/tasks"
    for task in tasks:
        assert get_task(task.task_id) is task
        assert (root / task.suite / f"{task.name}.py").is_file()
        assert task.critic_size > 52
    with pytest.raises(KeyError):
        get_task("libero_spatial/not_a_task")


def test_spatial_target_is_between_objects_and_not_distractor():
    # 两只黑碗必须有独立身份；between 任务的目标在盘子与 ramekin 之间。
    task = next(t for t in list_tasks("libero_spatial") if "between" in t.name)
    objects = {obj.name: obj for obj in task.objects}
    left, target, right = (objects[name] for name in ("ramekin", "bowl_target", "plate"))

    # 随机化范围也不能破坏左右次序或把干扰碗当成目标。
    assert left.position[0] + left.xy_noise < target.position[0] - target.xy_noise
    assert target.position[0] + target.xy_noise < right.position[0] - right.xy_noise
    assert objects["bowl_distractor"].position != target.position
    assert task.goal.target == target.name


def validate_results(root: Path) -> None:
    for task in list_tasks():
        folder = root / task.task_id
        result = json.loads((folder / "result.json").read_text())
        assert result["status"] == "PASS" and result["task_id"] == task.task_id
        assert result["num_envs"] == 4 and result["critic_size"] == task.critic_size
        assert result["ppo"]["transitions"] == 128
        assert result["ppo"]["actor_update"] > 0 and result["ppo"]["critic_update"] > 0
        assert not result["lw_imported"]
        for check in ("partial_reset", "terminal_observation", "success_fixture"):
            assert result[check] == "PASS"
        for artifact in ("initial_state.json", "image.png", "wrist_image.png"):
            assert (folder / artifact).stat().st_size > 0
    print("Five task artifacts PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    validate_results(parser.parse_args().output)

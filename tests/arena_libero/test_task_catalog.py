import argparse
import json
from pathlib import Path

import pytest
import torch

from carrot_sim.arena_libero.predicates import (
    inside_drawer_and_closed,
    placed_beside,
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


def test_catalog_has_all_registered_task_files():
    # 上游五个 suite 共 131 个注册 case；每个都要有独立模块与源类身份。
    tasks = list_tasks()
    counts = {
        "libero_spatial": 10,
        "libero_object": 10,
        "libero_goal": 11,
        "libero_10": 10,
        "libero_90": 90,
    }
    assert len(tasks) == 131
    for suite, count in counts.items():
        assert len(list_tasks(suite)) == count
    assert len({task.source_class for task in tasks}) == 131
    assert all(task.source_class for task in tasks)
    snapshot = json.loads(Path(__file__).with_name("task_sources.json").read_text())
    expected = {case["task_id"]: case["source_class"] for case in snapshot["cases"]}
    assert len(snapshot["cases"]) == len(expected) == 131
    assert {task.task_id: task.source_class for task in tasks} == expected

    # 注册名与文件路径保持一致，未知任务必须明确失败。
    root = Path(__file__).parents[2] / "src/carrot_sim/arena_libero/tasks"
    for task in tasks:
        assert get_task(task.task_id) is task
        assert (root / task.suite / f"{task.name}.py").is_file()
        assert task.critic_size >= 26
    with pytest.raises(KeyError):
        get_task("libero_spatial/not_a_task")


def test_beside_uses_world_directions_and_rejects_vertical_or_distant_objects():
    # LW 的 left/right 对应世界 +X/-X；不能让空中或远处同方向物体成功。
    positions = torch.tensor(
        [[0.16, 0, 0], [-0.16, 0, 0], [0, -0.16, 0], [0.16, 0, 0.5], [1.0, 0, 0]]
    )
    size = (0.05, 0.05, 0.03)

    # 同一 batch 分别检查方向、距离与高度，避免跨环境广播和方向颠倒。
    left = placed_beside(positions, size, size, "left", (0.001, 0.1), 0.25)
    right = placed_beside(positions, size, size, "right", (0.001, 0.1), 0.25)
    front = placed_beside(positions, size, size, "front", (0.001, 0.1), 0.25)
    assert left.tolist() == [True, False, False, False, False]
    assert right.tolist() == [False, True, False, False, False]
    assert front.tolist() == [False, False, True, False, False]


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


def test_middle_shelf_goal_contains_objects_resting_on_its_board():
    task = get_task(
        "libero_90/L90S4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf"
    )
    goal = task.goal
    lower = goal.center[2] - goal.half_size[2]
    upper = goal.center[2] + goal.half_size[2]
    board_top = -0.03319450095295906 * 1.2
    assert lower < board_top
    assert upper == pytest.approx(0.096 * 1.2)


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
    print(f"All {len(list_tasks())} task artifacts PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    validate_results(parser.parse_args().output)

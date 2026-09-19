from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


def _check_results(
    path: Path, suite: str, task_id: str, trials: int, minimum: int
) -> tuple[int, int]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    task_ids = range(10) if task_id == "all" else [int(task_id)]
    expected = {(task, episode) for task in task_ids for episode in range(trials)}
    assert all(record["suite"] == suite for record in records), suite
    assert all(type(record["success"]) is bool for record in records)
    keys = {(record["task_id"], record["episode_idx"]) for record in records}
    assert len(records) == len(expected), (len(records), len(expected))
    assert keys == expected, keys ^ expected
    successes = sum(record["success"] for record in records)
    assert successes >= minimum, (successes, minimum)
    return successes, len(expected)


def test_complete_task_results(tmp_path: Path) -> None:
    # 验证单 task 的每个 episode 恰好出现一次，成功数达到门槛才放行。
    # 构造两条不同 init state 的记录，覆盖 episode 索引和布尔成功字段。
    result_path = tmp_path / "episodes.jsonl"
    result_path.write_text(
        "\n".join(
            json.dumps(
                {"suite": "libero_spatial", "task_id": 0, "episode_idx": index, "success": True}
            )
            for index in range(2)
        )
    )

    # 校验完整性和成功数，防止缺记录或重复记录被当作有效 benchmark。
    assert _check_results(result_path, "libero_spatial", "0", 2, 2) == (2, 2)


def test_duplicate_episode_is_rejected(tmp_path: Path) -> None:
    # 验证同一 episode 重复写入时不能用记录总数冒充完整覆盖。
    # 两条记录都指向 episode 0，故缺少 episode 1。
    result_path = tmp_path / "episodes.jsonl"
    record = {"suite": "libero_spatial", "task_id": 0, "episode_idx": 0, "success": True}
    result_path.write_text("\n".join(json.dumps(record) for _ in range(2)))

    # 即使记录条数达到预期，也必须因唯一键集合不符而失败。
    with pytest.raises(AssertionError):
        _check_results(result_path, "libero_spatial", "0", 2, 2)


def test_full_suite_checks_all_ten_tasks(tmp_path: Path) -> None:
    # 验证全 suite 模式必须覆盖十个 task，不能只统计某一个 task 的成功率。
    # 每个 task 构造两个 episode，缩短单测同时保留完整 task 维度。
    result_path = tmp_path / "episodes.jsonl"
    result_path.write_text(
        "\n".join(
            json.dumps(
                {"suite": "libero_goal", "task_id": task, "episode_idx": index, "success": True}
            )
            for task in range(10)
            for index in range(2)
        )
    )

    # 全部键完整时通过；超过实际成功数的门槛必须失败。
    assert _check_results(result_path, "libero_goal", "all", 2, 20) == (20, 20)
    with pytest.raises(AssertionError):
        _check_results(result_path, "libero_goal", "all", 2, 21)


def main() -> None:
    path, suite, task_id, trials, minimum = sys.argv[1:]
    successes, total = _check_results(
        path=Path(path), suite=suite, task_id=task_id, trials=int(trials), minimum=int(minimum)
    )
    print(f"E2E_OK successes={successes}/{total} minimum={minimum} result={path}")


if __name__ == "__main__":
    main()

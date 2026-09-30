import json

import numpy as np
import pytest

from examples.so101_real.config import JOINT_NAMES
from examples.so101_real.report import write_report


def test_report_compares_only_completed_valid_frames(tmp_path):
    # 已保存的完整预测不等于实际消费动作；报告只统计成功完成的有效帧。
    np.savez_compressed(
        tmp_path / "chunk_000000.npz",
        predicted=np.array([[2] * 6, [99] * 6], dtype=np.float32),
        reference=np.array([[1] * 6, [0] * 6], dtype=np.float32),
        frame=np.array(5), planned_steps=np.array(2),
    )
    # 单位只能来自策略握手；外部degree角度和百分点夹爪。
    metadata = {"event": "metadata", "metadata": {
        "joint_units": "degrees", "gripper_units": "percentage_points",
    }}
    (tmp_path / "events.jsonl").write_text(
        json.dumps(metadata) + "\n" + json.dumps({"event": "action", "chunk": 0}) + "\n",
    )

    # 模拟第二步未完成，检查图表和指标都能由持久化日志重新生成。
    write_report(tmp_path)
    result = json.loads((tmp_path / "comparison.json").read_text())

    # 误差只能来自第一步，不能计入第二步巨大误差。
    assert result["compared_frames"] == 1
    assert result["joint_mae"] == dict.fromkeys(JOINT_NAMES, 1.0)
    assert result["joint_units"] == "degrees"
    assert result["gripper_units"] == "percentage_points"
    assert (tmp_path / "actions.png").stat().st_size > 0


def test_report_rejects_actions_without_unit_metadata(tmp_path):
    # 有动作但没有单位声明时必须失败，避免给单位不明的轨迹生成指标。
    np.savez_compressed(
        tmp_path / "chunk_000000.npz", predicted=np.ones((1, 6), dtype=np.float32),
        frame=np.array(0), planned_steps=np.array(1),
    )
    (tmp_path / "events.jsonl").write_text(json.dumps({"event": "action", "chunk": 0}) + "\n")

    # 不生成单位不明的指标或图片，错误需指向日志元数据。
    with pytest.raises(AssertionError, match="unit metadata"):
        write_report(tmp_path)
    assert not (tmp_path / "comparison.json").exists()

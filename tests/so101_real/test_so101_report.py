import json

import numpy as np

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
    # 动作消费事件足够生成报告，无需任何额外metadata声明。
    (tmp_path / "events.jsonl").write_text(
        json.dumps({"event": "action", "chunk": 0}) + "\n",
    )

    # 模拟第二步未完成，检查图表和指标都能由持久化日志重新生成。
    write_report(tmp_path)
    result = json.loads((tmp_path / "comparison.json").read_text())

    # 误差只能来自第一步，不能计入第二步巨大误差。
    assert result["compared_frames"] == 1
    assert result["joint_mae"] == dict.fromkeys(JOINT_NAMES, 1.0)
    assert set(result) == {"compared_frames", "joint_mae"}
    assert (tmp_path / "actions.png").stat().st_size > 0


def test_report_without_reference_or_metadata(tmp_path):
    # 只有预测与已执行事件的日志也能绘图，不能依赖metadata或不存在的示教动作。
    np.savez_compressed(
        tmp_path / "chunk_000000.npz", predicted=np.ones((1, 6), dtype=np.float32),
        frame=np.array(0), planned_steps=np.array(1),
    )
    (tmp_path / "events.jsonl").write_text(json.dumps({"event": "action", "chunk": 0}) + "\n")

    # 无reference只保存轨迹，比较帧数为0，不捏造误差指标。
    write_report(tmp_path)
    assert json.loads((tmp_path / "comparison.json").read_text()) == {"compared_frames": 0}
    assert (tmp_path / "actions.png").stat().st_size > 0

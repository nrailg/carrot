import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.robots.so_follower import SO101Follower

from carrot.data.dataset_spec import SFTDatasetSpec
from carrot.models.pi05.transforms import Normalize
from examples.so101_real.actions import load_action_limits
from recipes.pi05_sft_so101_state_jitter import augmentation


@pytest.fixture
def calibration_path(tmp_path: Path) -> Path:
    # 单一 LeRobot 原始标定文件，测试不依赖个人机器缓存或额外角度常量。
    names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    ranges = [(777, 3325), (916, 3333), (801, 3014), (980, 3178), (0, 4095), (1371, 2828)]
    path = tmp_path / "arm.json"
    path.write_text(json.dumps({
        name: {"id": index, "drive_mode": 0, "homing_offset": 0,
               "range_min": ranges[index - 1][0], "range_max": ranges[index - 1][1]}
        for index, name in enumerate(names, 1)
    }))
    return path


@pytest.fixture
def bounds(calibration_path: Path) -> list[list[float]]:
    # SDK 离线换算后取前五轴，gripper 不参与角度增强。
    lower, upper = load_action_limits(calibration_path)
    return np.stack((lower[:5], upper[:5]), axis=1).tolist()


def test_state_jitter_preserves_targets_and_raw_source(bounds: list[list[float]]) -> None:
    # 验证增强只修改五个角度输入，避免污染标签、gripper或底层缓存样本。
    sample = {"observation/state": np.array([10, -70, 90, 60, -4, 0.5], np.float32),
              "actions": np.ones((10, 6), np.float32), "action_is_pad": np.zeros(10, bool)}
    source_state = sample["observation/state"].copy()
    dataset = augmentation.StateJitterDataset([sample], 3.0, bounds)

    # 重复访问同一帧必须重新采样，正负两侧和范围都要被覆盖。
    states = np.stack([dataset[0]["observation/state"] for _ in range(256)])
    delta = states[:, :5] - source_state[:5]
    assert np.max(np.abs(delta)) <= 3.0
    assert np.all(delta.min(0) < -2) and np.all(delta.max(0) > 2)
    assert len(np.unique(states, axis=0)) == 256

    # 输出标签保持原对象；写入新state不能修改原始state或gripper。
    result = dataset[0]
    np.testing.assert_array_equal(sample["observation/state"], source_state)
    np.testing.assert_array_equal(states[:, 5], np.full(256, source_state[5]))
    assert result["actions"] is sample["actions"]
    assert result["action_is_pad"] is sample["action_is_pad"]
    assert len(dataset) == 1


def test_factory_preserves_stats_and_jitter_precedes_normalization(
    monkeypatch, calibration_path: Path,
) -> None:
    # 验证归一化统计来自干净数据，且度数扰动在归一化之前发生。
    sample = {"observation/state": np.full(6, 50, np.float32),
              "actions": np.full((10, 6), 20, np.float32)}
    stats = {"q01": np.zeros(6), "q99": np.full(6, 100)}
    spec = SFTDatasetSpec([sample], None, stats, stats, ("wrist",), embodiment="so101")
    def clean_factory(**kwargs: Any) -> SFTDatasetSpec:
        # 增强专用参数不能泄漏给原始数据工厂。
        assert kwargs == {"repo_id": "test/so101"}
        return spec

    monkeypatch.setattr(augmentation, "build_so101_dataset", clean_factory)
    monkeypatch.setattr(np.random, "uniform", lambda *args, **kwargs: np.full(5, 3.0))

    # 用确定的3度扰动检查端点，不能被解释成normalized空间的3个单位。
    wrapped = augmentation.build_dataset(
        state_jitter_degrees=3.0, state_jitter_calibration_path=str(calibration_path),
        repo_id="test/so101",
    )
    raw = wrapped.dataset[0]
    transformed = Normalize({"state": stats, "actions": stats}, True,
                            {"state": 6, "actions": 6})(raw | {"state": raw["observation/state"]})

    # Stats和除dataset外的契约保持相同；action归一化不受state增强影响。
    assert replace(wrapped, dataset=spec.dataset) == spec
    assert wrapped.state_stats is spec.state_stats and wrapped.action_stats is spec.action_stats
    np.testing.assert_array_equal(raw["observation/state"], [53, 53, 53, 53, 53, 50])
    np.testing.assert_allclose(transformed["state"][:5], 0.06, atol=2e-7)
    np.testing.assert_allclose(transformed["actions"], -0.6, atol=2e-7)


@pytest.mark.parametrize("direction", [-1, 1])
def test_state_jitter_clamps_at_calibrated_bounds(
    monkeypatch, direction: int, bounds: list[list[float]],
) -> None:
    # 边界附近的合法state加噪声会越界；必须clamp五轴，且不修改标签和gripper。
    limits = np.asarray(bounds, dtype=np.float32)
    edge = limits[:, 0 if direction == -1 else 1]
    state = np.append(edge - direction * 0.1, np.float32(150)).astype(np.float32)
    sample = {"observation/state": state, "actions": np.full((10, 6), 200, np.float32),
              "action_is_pad": np.zeros(10, bool)}
    original = state.copy()
    monkeypatch.setattr(np.random, "uniform", lambda *args, **kwargs: np.full(5, direction * 3.0))

    # 确定的向外扰动覆盖上/下界；结果按原始degrees裁剪，不对动作label做裁剪。
    result = augmentation.StateJitterDataset([sample], 3.0, bounds)[0]
    np.testing.assert_array_equal(result["observation/state"][:5], edge)
    assert result["observation/state"].dtype == np.float32
    assert result["observation/state"][5] == 150
    np.testing.assert_array_equal(sample["observation/state"], original)
    assert result["actions"] is sample["actions"]
    assert result["action_is_pad"] is sample["action_is_pad"]


@pytest.mark.parametrize("bounds", [[[-1, 1]] * 4, [[1, -1]] * 5, [[0, np.inf]] * 5])
def test_state_jitter_rejects_invalid_bounds(bounds: list[list[float]]) -> None:
    # 缺轴、上下界倒置及非有限配置应立即失败，不能静默用错关节限制。
    with pytest.raises(AssertionError, match="bounds"):
        augmentation.StateJitterDataset([], 3.0, bounds)


def test_jitter_recipes_share_calibration_path() -> None:
    # 所有使用同一增强工厂的 recipe 只保留文件路径，不能再复制五轴角度表。
    root = Path(__file__).resolve().parents[2] / "recipes"
    paths = set()
    for name in ("pi05_sft_so101_state_jitter", "pi05_sft_so101_state_jitter_matched_fit",
                 "pi05_sft_so101_state_only_rollout"):
        config = yaml.safe_load((root / name / "train.yaml").read_text())
        kwargs = config["dataset"]["factory_kwargs"]
        assert "state_jitter_bounds" not in kwargs
        paths.add(kwargs["state_jitter_calibration_path"])

    # 仅引用同一份 LeRobot JSON；资源实存和 Mac/GPU SHA 在上机前单独核验。
    assert len(paths) == 1
    assert Path(paths.pop()).name == "my_awesome_follower_arm.json"


def test_factory_reads_updated_calibration_without_hardware(
    monkeypatch, calibration_path: Path,
) -> None:
    # 修改唯一标定文件应直接改变增强限位，不必同时改多个 YAML；SDK 不能打开串口。
    sample = {"observation/state": np.zeros(6, np.float32)}
    spec = SFTDatasetSpec([sample], None, None, None, ("wrist",), embodiment="so101")
    monkeypatch.setattr(augmentation, "build_so101_dataset", lambda **kwargs: spec)
    def forbid_connect(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("calibration conversion must not connect hardware")
    monkeypatch.setattr(SO101Follower, "connect", forbid_connect)
    monkeypatch.setattr(FeetechMotorsBus, "connect", forbid_connect)

    # 连续构造两次工厂，第二次读取变更后的 raw 端点而不是缓存的角度常量。
    first = augmentation.build_dataset(
        state_jitter_degrees=3.0, state_jitter_calibration_path=str(calibration_path),
    )
    calibration = json.loads(calibration_path.read_text())
    calibration["shoulder_lift"]["range_min"] -= 16
    calibration["shoulder_lift"]["range_max"] += 16
    calibration_path.write_text(json.dumps(calibration))
    second = augmentation.build_dataset(
        state_jitter_degrees=3.0, state_jitter_calibration_path=str(calibration_path),
    )

    # 只有肩抬变化，中心不变；换算允许 float32 一个 ULP，不影响其他轴/底层样本。
    np.testing.assert_array_equal(first.dataset.upper[[0, 2, 3, 4]],
                                  second.dataset.upper[[0, 2, 3, 4]])
    assert second.dataset.upper[1] > first.dataset.upper[1]
    assert second.dataset.lower[1] == -second.dataset.upper[1]
    np.testing.assert_allclose(second.dataset.upper[1] - first.dataset.upper[1],
                               16 * 360 / 4095, rtol=0, atol=2e-5)


def test_factory_rejects_missing_calibration_before_dataset_io(monkeypatch, tmp_path: Path) -> None:
    # GPU 缺标定文件时必须在读取训练数据前失败，不能退回写死的 bounds。
    def forbid_data(**kwargs: Any) -> None:
        raise AssertionError("dataset must not load before calibration validation")
    monkeypatch.setattr(augmentation, "build_so101_dataset", forbid_data)

    # 明确报出缺失文件；不创建标定、不下载，也不连接硬件。
    with pytest.raises(AssertionError, match="missing SO101 calibration"):
        augmentation.build_dataset(
            state_jitter_degrees=3.0, state_jitter_calibration_path=str(tmp_path / "missing.json"),
        )

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from carrot.data.dataset_spec import SFTDatasetSpec
from carrot.models.pi05.transforms import Normalize
from recipes.pi05_sft_so101_state_jitter import augmentation

BOUNDS = [
    [-112.0, 112.0],
    [-106.24175824175825, 106.24175824175825],
    [-97.27472527472527, 97.27472527472527],
    [-96.61538461538461, 96.61538461538461],
    [-180.0, 180.0],
]


def test_state_jitter_preserves_targets_and_raw_source() -> None:
    # 验证增强只修改五个角度输入，避免污染标签、gripper或底层缓存样本。
    sample = {"observation/state": np.array([10, -70, 90, 60, -4, 0.5], np.float32),
              "actions": np.ones((10, 6), np.float32), "action_is_pad": np.zeros(10, bool)}
    source_state = sample["observation/state"].copy()
    dataset = augmentation.StateJitterDataset([sample], 3.0, BOUNDS)

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


def test_factory_preserves_stats_and_jitter_precedes_normalization(monkeypatch) -> None:
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
        state_jitter_degrees=3.0, state_jitter_bounds=BOUNDS, repo_id="test/so101"
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
def test_state_jitter_clamps_at_calibrated_bounds(monkeypatch, direction: int) -> None:
    # 边界附近的合法state加噪声会越界；必须clamp五轴，且不修改标签和gripper。
    limits = np.asarray(BOUNDS, dtype=np.float32)
    edge = limits[:, 0 if direction == -1 else 1]
    state = np.append(edge - direction * 0.1, np.float32(150)).astype(np.float32)
    sample = {"observation/state": state, "actions": np.full((10, 6), 200, np.float32),
              "action_is_pad": np.zeros(10, bool)}
    original = state.copy()
    monkeypatch.setattr(np.random, "uniform", lambda *args, **kwargs: np.full(5, direction * 3.0))

    # 确定的向外扰动覆盖上/下界；结果按原始degrees裁剪，不对动作label做裁剪。
    result = augmentation.StateJitterDataset([sample], 3.0, BOUNDS)[0]
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


def test_jitter_recipes_use_current_follower_limits() -> None:
    # 三个使用同一wrapper的recipe均需显式绑定当前机器标定，避免工厂签名修改后漏配。
    root = Path(__file__).resolve().parents[2] / "recipes"
    for name in ("pi05_sft_so101_state_jitter", "pi05_sft_so101_state_jitter_matched_fit",
                 "pi05_sft_so101_state_only_rollout"):
        config = yaml.safe_load((root / name / "train.yaml").read_text())
        np.testing.assert_array_equal(
            config["dataset"]["factory_kwargs"]["state_jitter_bounds"], BOUNDS
        )

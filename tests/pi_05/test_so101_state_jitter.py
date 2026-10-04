from dataclasses import replace

import numpy as np

from carrot.data.dataset_spec import SFTDatasetSpec
from carrot.models.pi05.transforms import Normalize
from recipes.pi05_sft_so101_state_jitter import augmentation


def test_state_jitter_preserves_targets_and_raw_source() -> None:
    # 验证增强只修改五个角度输入，避免污染标签、gripper或底层缓存样本。
    sample = {"observation/state": np.array([10, -70, 90, 60, -4, 0.5], np.float32),
              "actions": np.ones((10, 6), np.float32), "action_is_pad": np.zeros(10, bool)}
    source_state = sample["observation/state"].copy()
    dataset = augmentation.StateJitterDataset([sample], 3.0)

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
    monkeypatch.setattr(augmentation, "build_so101_dataset", lambda **kwargs: spec)
    monkeypatch.setattr(np.random, "uniform", lambda *args, **kwargs: np.full(5, 3.0))

    # 用确定的3度扰动检查端点，不能被解释成normalized空间的3个单位。
    wrapped = augmentation.build_dataset(state_jitter_degrees=3.0)
    raw = wrapped.dataset[0]
    transformed = Normalize({"state": stats, "actions": stats}, True,
                            {"state": 6, "actions": 6})(raw | {"state": raw["observation/state"]})

    # Stats和除dataset外的契约保持相同；action归一化不受state增强影响。
    assert replace(wrapped, dataset=spec.dataset) == spec
    assert wrapped.state_stats is spec.state_stats and wrapped.action_stats is spec.action_stats
    np.testing.assert_array_equal(raw["observation/state"], [53, 53, 53, 53, 53, 50])
    np.testing.assert_allclose(transformed["state"][:5], 0.06, atol=2e-7)
    np.testing.assert_allclose(transformed["actions"], -0.6, atol=2e-7)

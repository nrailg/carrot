import numpy as np
import pytest

from carrot.data.mixture_schedule import make_schedule, valid_lengths


@pytest.mark.parametrize("mixed", [False, True])
def test_each_source_is_exhausted_before_reshuffling(mixed: bool) -> None:
    # 每个来源必须完整无放回遍历再重洗；不同大小的数据集不能共享错误的轮次边界。
    # 互质大小让遍历边界落在batch内部，并同时覆盖多轮和最后的不完整轮次。
    sources = (np.arange(17) % 10 + 1, np.arange(29) % 10 + 1)
    plan = make_schedule(*sources, mixed=mixed, steps=3)
    repeat = make_schedule(*sources, mixed=mixed, steps=3)
    # 同种子逐字段可复现；样本长度必须与实际索引绑定，不能沿用洗牌前的长度。
    for key in plan:
        np.testing.assert_array_equal(plan[key], repeat[key])
    for domain in range(2 if mixed else 1):
        selected = plan["domain"] == domain
        indices = plan["source_index"][selected]
        size = len(sources[domain])
        np.testing.assert_array_equal(plan["valid_length"][selected], sources[domain][indices])
        # 每个完整轮次恰好覆盖全体样本；末尾不足一轮时也不能重复。
        for start in range(0, len(indices), size):
            chunk = indices[start : start + size]
            assert len(np.unique(chunk)) == len(chunk)
            if len(chunk) == size:
                np.testing.assert_array_equal(np.sort(chunk), np.arange(size))
        # 下一轮须重新打乱，而不是无限重播第一轮的固定顺序。
        assert not np.array_equal(indices[:size], indices[size : 2 * size])


def test_experiment_budget_has_no_repeated_start_frames() -> None:
    # 本轮200步预算小于各数据集大小，应完全没有重复起点并保持配对曝光。
    # 使用真实帧数，避免只在跨多轮的小fixture上验证而漏掉实际实验边界。
    sources = (np.full(273465, 10), np.full(148165, 10))
    pure = make_schedule(*sources, mixed=False)
    mixed = make_schedule(*sources, mixed=True)
    # 纯LIBERO和混合各来源的整段抽样都不得重复；混合LIBERO等于纯路线的前缀。
    assert len(np.unique(pure["source_index"])) == 51200
    for domain in (0, 1):
        indices = mixed["source_index"][mixed["domain"] == domain]
        assert len(indices) == len(np.unique(indices)) == 25600
    np.testing.assert_array_equal(
        pure["source_index"][:25600], mixed["source_index"][mixed["domain"] == 0]
    )


def test_exact_source_quotas_and_paired_exposure() -> None:
    # 验证分卡和累积后的每步配额，并保护 L100/Mix200 的 LIBERO 曝光一致性。
    # 使用不同轨迹长度覆盖尾帧监督数量，避免只有全长 chunk 的退化情况。
    lengths = np.arange(1, 11).repeat(3)
    pure = make_schedule(lengths, lengths[::-1], mixed=False)
    mixed = make_schedule(lengths, lengths[::-1], mixed=True)
    # 完整批量必须128/128；每张卡跨8次micro累积后各取16个。
    domains = mixed["domain"].reshape(200, 256)
    assert np.all(domains.sum(axis=1) == 128)
    for rank in range(8):
        assert np.all(domains[:, rank::8].sum(axis=1) == 16)
    # 仅比较相同的LIBERO曝光序列，不要求两路数据在同一个优化步出现。
    assert np.array_equal(pure["source_index"][:25600], mixed["source_index"][mixed["domain"] == 0])


@pytest.mark.parametrize("mixed", [False, True])
def test_normalization_survives_microbatch_and_rank_partition(mixed: bool) -> None:
    # 验证目标按域有效时间步归一化，DP与micro切分不改变梯度系数。
    # 用每条样本不同的误差模拟非均匀监督，避免常数loss掩盖归一化错误。
    lengths = np.arange(1, 11)
    plan = make_schedule(lengths, lengths[::-1], mixed=mixed, steps=1)
    errors = np.random.default_rng(5).uniform(size=(256, 10))
    valid = np.arange(10)[None] < plan["valid_length"][:, None]
    sums = (errors * valid).sum(axis=1)
    # 直接计算完整global batch的定义，作为独立oracle。
    expected = 0.0
    for domain in range(2 if mixed else 1):
        selected = plan["domain"] == domain
        expected += (0.5 if mixed else 1.0) * sums[selected].sum() / valid[selected].sum()
    # 模拟8卡、每卡micro4与8次累积的实际缩放顺序。
    actual = 0.0
    for rank in range(8):
        indices = np.arange(rank, 256, 8).reshape(8, 4)
        actual += sum((sums[i] * plan["loss_weight"][i]).mean() / 8 for i in indices) / 8
    assert actual == pytest.approx(expected, rel=1e-7)


def test_action_lengths_end_at_episode_boundary() -> None:
    # 防止动作目标跨episode；最后一帧必须只监督当前动作。
    # 相邻episode故意小于horizon，暴露越界与off-by-one。
    ids = np.array([7, 7, 7, 8, 8])
    result = valid_lengths(ids)
    # episode编号可不从0开始，但必须连续存储。
    np.testing.assert_array_equal(result, [3, 2, 1, 2, 1])
    with pytest.raises(AssertionError, match="noncontiguous"):
        valid_lengths(np.array([7, 8, 7]))

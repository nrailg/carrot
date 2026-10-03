import pytest
from torch.utils.data import DataLoader, DistributedSampler

from carrot.data.dataset_spec import SFTDatasetSpec
from recipes.pi05_sft_so101_knock_down_the_cylinder_overfit import single_frame


def test_single_frame_keeps_targets_stats_and_distributed_batches(monkeypatch) -> None:
    # 固定一个源样本，所有rank必须只读这一帧，同时保留源stats和可用batch。
    # 用不同源值区分误选帧，metadata保持引用以检测不必要的统计量重算。
    spec = SFTDatasetSpec(
        dataset=list(range(264)), collate_fn=None,
        state_stats={"q01": [1.0]}, action_stats={"q99": [2.0]},
        image_keys=("observation.images.wrist",), embodiment="so101",
    )
    monkeypatch.setattr(single_frame, "build_so101_dataset", lambda **kwargs: spec)

    # 虚拟64样本支持8rank × micro4 × GAS2，唯一源帧仍为53。
    actual = single_frame.build_dataset(frame_index=53, repeat_count=64)
    assert len(actual.dataset) == 64
    assert actual.state_stats is spec.state_stats and actual.action_stats is spec.action_stats
    assert actual.image_keys == spec.image_keys and actual.embodiment == spec.embodiment

    # 与生产相同的drop_last设置不能产生空loader或混入其他源样本。
    for rank in range(8):
        sampler = DistributedSampler(actual.dataset, num_replicas=8, rank=rank, drop_last=True)
        batches = list(DataLoader(actual.dataset, batch_size=4, sampler=sampler, drop_last=True))
        assert len(batches) == 2
        assert all(batch.tolist() == [53] * 4 for batch in batches)


@pytest.mark.parametrize("frame_index, repeat_count", [(-1, 64), (264, 64), (53, 0)])
def test_single_frame_rejects_invalid_selection(monkeypatch, frame_index, repeat_count) -> None:
    # 越界源帧或空虚拟数据集必须在启动训练前失败。
    # 构造合法源长度，隔离frame_index/repeat_count校验。
    spec = SFTDatasetSpec(
        dataset=list(range(264)), collate_fn=None, state_stats={}, action_stats={},
        image_keys=("observation.images.wrist",), embodiment="so101",
    )
    monkeypatch.setattr(single_frame, "build_so101_dataset", lambda **kwargs: spec)

    # 错误配置不得静默回退到全量训练。
    with pytest.raises(AssertionError):
        single_frame.build_dataset(frame_index=frame_index, repeat_count=repeat_count)

import json
from typing import override

import numpy as np
import pytest
import torch
from torch import nn

from carrot.data.libero_draw import DrawDataset, ScheduledDataset
from carrot.data.mixture_schedule import make_schedule
from carrot.models.pi05.loss_fn import Pi05SFTLossFn


def write_episode(root) -> None:
    episode = root / "episode-0000"
    episode.mkdir(parents=True)
    (episode / "result.json").write_text(
        json.dumps({"success": True, "picks": 3, "returns": 3, "seed": 200000})
    )
    arrays = {
        "image": np.zeros((4, 8, 8, 3), np.uint8),
        "wrist_image": np.zeros((4, 8, 8, 3), np.uint8),
        "state": np.zeros((4, 8), np.float32),
        "prompt": np.array(["pick red"] * 4),
        "phase": np.array([0, 0, 1, 1]),
        "actions": np.ones((4, 7), np.float32),
    }
    for name, values in arrays.items():
        np.save(episode / f"{name}.npy", values)


def test_draw_adapter_masks_phase_boundaries(tmp_path) -> None:
    # 保护动作监督不跨子任务；补齐部分不能被当作有效零动作。
    # 两个相邻phase只有两帧，确保horizon10会触及边界。
    write_episode(tmp_path)
    dataset = DrawDataset(str(tmp_path))
    sample = dataset[1]
    # phase末帧仅保留一个动作，后九个padding同时清零和置mask。
    assert dataset.lengths.tolist() == [2, 1, 2, 1]
    assert sample["action_is_pad"].tolist() == [False] + [True] * 9
    assert np.all(sample["actions"][0] == 1) and np.all(sample["actions"][1:] == 0)


def test_scheduled_adapter_rejects_wrong_mask(tmp_path) -> None:
    # 清单与数据边界不一致必须报错，防止错误分母污染整个global batch。
    # 用同一小数据代替两个来源，此测试只覆盖索引与边界契约。
    write_episode(tmp_path / "data")
    data = DrawDataset(str(tmp_path / "data"))
    schedule = make_schedule(data.lengths, data.lengths, mixed=True, steps=1)
    schedule["valid_length"][0] = 10
    np.savez(tmp_path / "samples.npz", **schedule)
    # 错误不允许静默修正，因为权重已经根据旧长度计算。
    dataset = ScheduledDataset(data, data, str(tmp_path / "samples.npz"))
    with pytest.raises(AssertionError, match="action boundary"):
        dataset[0]


class ScalarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(2.0))

    @override
    def forward(self, observation, actions, noise, time):
        return (self.value - actions).square()


class FixtureLoss(Pi05SFTLossFn):
    def __init__(self):
        pass

    @override
    def prepare_inputs(self, model, batch):
        return None, batch["actions"], ~batch["action_is_pad"]


def test_actual_weighted_loss_gradient_matches_global_oracle() -> None:
    # 直接调用生产loss，验证既不漏乘权重，也不重复除有效步数。
    # 两个域的长度和误差不同，独立oracle可辨别任何等权误用。
    model = ScalarModel()
    actions = torch.tensor([0.0, 1.0]).view(2, 1, 1).expand(2, 10, 7)
    batch = {
        "actions": actions,
        "action_is_pad": torch.arange(10)[None] >= torch.tensor([1, 3])[:, None],
        "loss_weight": torch.tensor([1.0, 1 / 3]),
        "sample_domain": torch.tensor([0, 1]),
    }
    # 两域均值分别为4和1，均分后的loss=2.5，导数=3。
    loss, metrics = FixtureLoss()(model, batch)
    loss.backward()
    assert float(loss) == pytest.approx(2.5)
    assert float(model.value.grad) == pytest.approx(3.0)
    assert float(metrics["loss_libero"] + metrics["loss_draw"]) == pytest.approx(2.5)

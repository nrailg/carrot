from types import MethodType

import numpy as np
import pytest
import torch

from carrot.models.pi05.loss_fn import Pi05SFTLossFn
from recipes.pi05_sft_so101_fit_validation.fit import DropVision, FitLoss, fixed_noise


class _ProbeModel:
    def __init__(self) -> None:
        self.inputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def __call__(self, observation, actions, *, noise, time):
        self.inputs.append((noise.clone(), time.clone()))
        return (noise - actions + time[:, None, None]).square()


def _source(actions: torch.Tensor, valid: torch.Tensor) -> Pi05SFTLossFn:
    source = Pi05SFTLossFn(None, state_stats={}, action_stats={}, image_keys=())
    source.prepare_inputs = MethodType(lambda self, model, batch: (None, actions, valid), source)
    return source


def test_random_fit_loss_matches_production(monkeypatch: pytest.MonkeyPatch) -> None:
    # 相同RNG状态下，recipe的随机noise/t和32维mask loss必须与生产实现完全一致。
    actions = torch.arange(2 * 10 * 32, dtype=torch.float32).reshape(2, 10, 32) / 1000
    valid = torch.ones(2, 10, dtype=torch.bool)
    valid[1, 7:] = False
    source = _source(actions, valid)
    fit = FitLoss(source, None)
    monkeypatch.setattr(fit, "prepare_inputs", source.prepare_inputs)
    model = _ProbeModel()

    # 重置随机状态，检测更换采样分布、loss维度或padding权重的意外差异。
    torch.manual_seed(730)
    expected, _ = source(model, {})
    torch.manual_seed(730)
    actual, _ = fit(model, {})
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for lhs, rhs in zip(model.inputs[0], model.inputs[1], strict=True):
        torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)


def test_fixed_noise_keeps_time_random(monkeypatch: pytest.MonkeyPatch) -> None:
    # 同帧固定latent不能无意固定t；所有batch行和验证入口必须使用同一噪声。
    source = _source(torch.zeros(4, 10, 32), torch.ones(4, 10, dtype=torch.bool))
    fit = FitLoss(source, 1053)
    monkeypatch.setattr(fit, "prepare_inputs", source.prepare_inputs)
    model = _ProbeModel()

    # 两次forward应只改变time，BF16量化后的latent需逐元素匹配验证生成器。
    fit(model, {})
    fit(model, {})
    expected = torch.from_numpy(fixed_noise(1053, 10, 32))
    for noise, time in model.inputs:
        torch.testing.assert_close(noise, expected[None].expand(4, 10, 32), rtol=0, atol=0)
        assert torch.all((time >= 0.001) & (time <= 1.0))
    assert not torch.equal(model.inputs[0][1], model.inputs[1][1])


def test_drop_vision_removes_tokens_without_mutating_sample() -> None:
    # 无视觉条件必须关闭attention mask，不能只把图片变黑或改动state/actions。
    image = np.ones((3, 224, 224), dtype=np.uint8)
    data = {"image": {"base_0_rgb": image, "left_wrist_0_rgb": image},
            "image_mask": {"base_0_rgb": False, "left_wrist_0_rgb": True},
            "state": np.arange(32), "actions": np.arange(320).reshape(10, 32)}

    # 输入共享引用保持原样；只改本次变换结果中的image与mask。
    actual = DropVision()(data)
    assert all(not image.any() for image in actual["image"].values())
    assert all(mask is False for mask in actual["image_mask"].values())
    assert data["image_mask"]["left_wrist_0_rgb"] is True and image.all()
    assert actual["state"] is data["state"] and actual["actions"] is data["actions"]

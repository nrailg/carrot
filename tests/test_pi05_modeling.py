from __future__ import annotations

from typing import Any

import torch
from torch import nn

from carrot.models.pi05.model import PI0Observation
from carrot.models.pi05.modeling import Pi05SFTLossFn


class _Tokenizer:
    def __call__(self, prompts: list[str], **_: Any) -> dict[str, torch.Tensor]:
        batch = len(prompts)
        return {
            "input_ids": torch.ones(batch, 200, dtype=torch.long),
            "attention_mask": torch.ones(batch, 200, dtype=torch.long),
        }


class _Policy(nn.Module):
    # 捕获 loss adapter 的调用参数，避免用真实大模型掩盖 batch contract 问题。
    max_state_dim = 32
    max_action_dim = 32

    def __init__(self) -> None:
        super().__init__()
        self.action_in_proj = nn.Linear(32, 4)
        self.seen: tuple[Any, ...] | None = None

    def forward(
        self,
        observation: PI0Observation,
        actions: torch.Tensor,
        noise: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        self.seen = observation, actions, noise, time
        return (actions - noise).square()


def _stats() -> dict[str, list[float]]:
    return {"q01": [-1.0] * 14, "q99": [1.0] * 14}


def test_pi05_batch_contract_and_padding_mask() -> None:
    # 验证 SFT loss 能把原始 batch 整理成官方 PI0Observation，并接受动作 padding mask。
    # Arrange：fake policy 只记录官方调用契约，loss adapter 使用 14 维归一化统计。
    native = _Policy()
    loss_fn = Pi05SFTLossFn(
        _Tokenizer(),
        state_stats=_stats(),
        action_stats=_stats(),
        image_keys=("high", "left", "right"),
        preprocess=None,
    )

    # loss adapter 应保持为普通 callable，避免被训练器误当作待并行化模型。
    assert not isinstance(loss_fn, nn.Module)

    # Arrange：不同图像尺寸和非对称 padding 用来暴露预处理、补维及 mask shape 错误。
    batch = {
        "high": torch.ones(2, 3, 480, 640),
        "left": torch.rand(2, 3, 16, 16),
        "right": torch.rand(2, 3, 16, 16),
        "observation.state": torch.zeros(2, 14),
        "action": torch.zeros(2, 50, 14),
        "action_is_pad": torch.tensor([[False] * 50, [False] + [True] * 49]),
        "task": ["pick bottle", "place cup"],
    }

    # Act：执行 loss，fake policy 记录传入的官方四参数调用契约。
    loss, metrics = loss_fn(native, batch)
    observation, noisy_actions, noise, time = native.seen

    # Assert：先检查 loss contract，再检查 observation 中各模态的布局和补维结果。
    assert loss.ndim == 0
    assert metrics["per_step_loss"].shape == (2, 50)
    images = list(observation.images.values())
    masks = list(observation.image_masks.values())
    tokens = observation.tokenized_prompt
    token_masks = observation.tokenized_prompt_mask
    state = observation.state

    # 大图应等比缩放并补黑边，三个相机最终都必须是模型要求的 NCHW 224 图像。
    assert [tuple(image.shape) for image in images] == [(2, 3, 224, 224)] * 3
    assert torch.all(images[0][..., 0, :] == -1)
    assert torch.all(images[0][..., 112, :] == 1)
    assert all(mask.dtype is torch.bool for mask in masks)

    # tokenizer 输出和 14 维机器人量必须分别满足固定 token 长度及官方 32 维接口。
    assert tokens.shape == token_masks.shape == (2, 200)
    assert state.shape == (2, 32)
    assert noisy_actions.shape == (2, 50, 32)
    assert time.shape == (2,)

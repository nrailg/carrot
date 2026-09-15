from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from carrot.models.pi05.loss_fn import Pi05SFTLossFn
from carrot.models.pi05.model import PI0Observation, PI0Policy


class _Tokenizer:
    def __call__(self, prompts: list[str], **_: Any) -> dict[str, torch.Tensor]:
        batch = len(prompts)
        return {
            "input_ids": torch.ones(batch, 200, dtype=torch.long),
            "attention_mask": torch.ones(batch, 200, dtype=torch.long),
        }


class _Policy(nn.Module):
    # 捕获 loss adapter 的调用参数，避免用真实大模型掩盖 batch contract 问题。
    def __init__(self) -> None:
        super().__init__()
        # 模拟 Diffusers ConfigMixin 暴露的只读属性访问接口。
        self.config = SimpleNamespace(action_dim=32)
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


def test_pi05_openpi_checkpoint_round_trip(tmp_path: Path) -> None:
    # 验证 PI0Policy 导出官方 OpenPI PyTorch 文件名与配置 schema，并能严格回读权重。
    # Arrange：dummy Gemma 保留完整模块和 tied-weight 关系，同时限制测试资源消耗。
    policy = PI0Policy(
        dtype="float32",
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=4,
        action_horizon=2,
        pi05=True,
        pytorch_compile_mode=None,
    )
    expected_action_in = policy.action_in_proj.weight.detach().clone()
    expected_language = (
        policy.paligemma_with_expert.paligemma.language_model.embed_tokens.weight[:4]
        .detach()
        .clone()
    )

    # Act：直接导出到 checkpoint 根目录，再走 model.safetensors 专用加载分支回读。
    policy.save_pretrained(tmp_path)
    restored = PI0Policy.from_pretrained(tmp_path)

    # Assert：文件布局和五个配置字段必须与 OpenPI 转换脚本的输出契约一致。
    assert (tmp_path / "model.safetensors").is_file()
    with (tmp_path / "config.json").open() as stream:
        config = json.load(stream)
    assert config == {
        "action_dim": 4,
        "action_horizon": 2,
        "paligemma_variant": "dummy",
        "action_expert_variant": "dummy",
        "precision": "float32",
    }

    # Assert：动作投影与 tied language embedding 均严格一致，避免导出时错误去重权重。
    torch.testing.assert_close(restored.action_in_proj.weight, expected_action_in, rtol=0, atol=0)
    torch.testing.assert_close(
        restored.paligemma_with_expert.paligemma.language_model.embed_tokens.weight[:4],
        expected_language,
        rtol=0,
        atol=0,
    )

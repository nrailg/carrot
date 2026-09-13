from __future__ import annotations

from typing import Any

import torch
from torch import nn

from carrot.models.pi05.modeling import Pi05SFTPolicy


class _Tokenizer:
    def __call__(self, prompts: list[str], **_: Any) -> dict[str, torch.Tensor]:
        batch = len(prompts)
        return {
            "input_ids": torch.ones(batch, 200, dtype=torch.long),
            "attention_mask": torch.ones(batch, 200, dtype=torch.long),
        }


class _Policy(nn.Module):
    max_state_dim = 32
    max_action_dim = 32

    def __init__(self) -> None:
        super().__init__()
        self.action_in_proj = nn.Linear(32, 4)
        self.seen: tuple[Any, ...] | None = None

    def forward(self, images, masks, tokens, token_masks, state, noisy_actions, time):
        self.seen = images, masks, tokens, token_masks, state, noisy_actions, time
        return noisy_actions * self.action_in_proj.weight.sum() * 0


def _stats() -> dict[str, list[float]]:
    return {"q01": [-1.0] * 14, "q99": [1.0] * 14}


def test_pi05_batch_contract_and_padding_mask() -> None:
    native = _Policy()
    policy = Pi05SFTPolicy(
        native,
        _Tokenizer(),
        state_stats=_stats(),
        action_stats=_stats(),
        image_keys=("high", "left", "right"),
        preprocess=None,
    )
    batch = {
        "high": torch.ones(2, 3, 480, 640),
        "left": torch.rand(2, 3, 16, 16),
        "right": torch.rand(2, 3, 16, 16),
        "observation.state": torch.zeros(2, 14),
        "action": torch.zeros(2, 50, 14),
        "action_is_pad": torch.tensor([[False] * 50, [False] + [True] * 49]),
        "task": ["pick bottle", "place cup"],
    }

    loss, metrics = policy(batch)
    images, masks, tokens, token_masks, state, noisy_actions, time = native.seen

    assert loss.ndim == 0
    assert metrics["per_step_loss"].shape == (2, 50)
    assert [tuple(image.shape) for image in images] == [(2, 3, 224, 224)] * 3
    assert torch.all(images[0][..., 0, :] == -1)
    assert torch.all(images[0][..., 112, :] == 1)
    assert all(mask.dtype is torch.bool for mask in masks)
    assert tokens.shape == token_masks.shape == (2, 200)
    assert state.shape == (2, 32)
    assert noisy_actions.shape == (2, 50, 32)
    assert time.shape == (2,)

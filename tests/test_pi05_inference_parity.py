from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05 import PI05Policy
from safetensors import safe_open

from carrot.models.pi05.model import PI0Policy
from tests.pi05_checkpoint_utils import lerobot_key_for_open_giga_key

# 测试环境可临时指定本地 Hub snapshot；测试不会访问或改写远端 checkpoint。
LEROBOT_CHECKPOINT = os.environ.get("CARROT_PI05_LEROBOT_CHECKPOINT")
pytestmark = pytest.mark.skipif(
    LEROBOT_CHECKPOINT is None or not torch.cuda.is_available(),
    reason="set CARROT_PI05_LEROBOT_CHECKPOINT and run on a CUDA host",
)


def _inputs(
    device: torch.device,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    generator = torch.Generator(device=device).manual_seed(20260914)
    images = [
        torch.rand((1, 3, 224, 224), generator=generator, device=device) * 2 - 1 for _ in range(3)
    ]
    image_masks = [torch.ones(1, dtype=torch.bool, device=device) for _ in images]
    tokens = torch.arange(1, 65, dtype=torch.long, device=device)[None, :]
    token_masks = torch.ones_like(tokens, dtype=torch.bool)
    state = torch.zeros((1, 32), dtype=torch.float32, device=device)
    noise = torch.randn((1, 50, 32), generator=generator, device=device)
    return images, image_masks, tokens, token_masks, state, noise


def _load_lerobot(device: torch.device) -> PI05Policy:
    config = PreTrainedConfig.from_pretrained(LEROBOT_CHECKPOINT)
    config.device = str(device)
    config.dtype = "float32"
    config.compile_model = False
    return PI05Policy.from_pretrained(LEROBOT_CHECKPOINT, config=config, strict=True).eval()


def _load_open_giga_from_lerobot(device: torch.device) -> PI0Policy:
    model = PI0Policy(pi05_enabled=True)
    target_keys = model.state_dict().keys()
    with safe_open(
        Path(LEROBOT_CHECKPOINT) / "model.safetensors", framework="pt", device="cpu"
    ) as checkpoint:
        state_dict = {
            target_key: checkpoint.get_tensor(lerobot_key_for_open_giga_key(target_key))
            for target_key in target_keys
        }
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def test_open_giga_model_matches_lerobot_denoising_and_sampling() -> None:
    device = torch.device("cuda")
    images, image_masks, tokens, token_masks, state, noise = _inputs(device)

    lerobot = _load_lerobot(device)
    lerobot_one_step = lerobot.model.sample_actions(
        images, image_masks, tokens, token_masks, noise=noise.clone(), num_steps=1
    )
    lerobot_actions = lerobot.model.sample_actions(
        images, image_masks, tokens, token_masks, noise=noise.clone(), num_steps=10
    )
    del lerobot
    torch.cuda.empty_cache()

    open_giga = _load_open_giga_from_lerobot(device)
    open_giga.num_steps = 1
    open_giga_one_step = open_giga.sample_actions(
        images, image_masks, tokens, token_masks, state, noise=noise.clone()
    )
    open_giga.num_steps = 10
    open_giga_actions = open_giga.sample_actions(
        images, image_masks, tokens, token_masks, state, noise=noise.clone()
    )

    torch.testing.assert_close(open_giga_one_step, lerobot_one_step, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(open_giga_actions, lerobot_actions, rtol=1e-2, atol=5e-3)

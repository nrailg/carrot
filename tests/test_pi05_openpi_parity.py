from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from carrot.models.pi05.model import PI0Policy

OPENPI_GOLDEN = os.environ.get("CARROT_PI05_OPENPI_GOLDEN")
OPEN_GIGA_CHECKPOINT = os.environ.get("CARROT_PI05_OPEN_GIGA_CHECKPOINT")
pytestmark = pytest.mark.skipif(
    OPENPI_GOLDEN is None or OPEN_GIGA_CHECKPOINT is None or not torch.cuda.is_available(),
    reason=(
        "set CARROT_PI05_OPENPI_GOLDEN and CARROT_PI05_OPEN_GIGA_CHECKPOINT "
        "and run on a CUDA host"
    ),
)


def _load_inputs(
    archive: np.lib.npyio.NpzFile, device: torch.device
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    images = [
        torch.from_numpy(image).permute(0, 3, 1, 2).contiguous().to(device)
        for image in archive["images"]
    ]
    image_masks = [torch.from_numpy(mask).to(device) for mask in archive["image_masks"]]
    tokens = torch.from_numpy(archive["tokens"]).to(device=device, dtype=torch.long)
    token_masks = torch.from_numpy(archive["token_masks"]).to(device)
    state = torch.from_numpy(archive["state"]).to(device)
    noise = torch.from_numpy(archive["noise"]).to(device)
    return images, image_masks, tokens, token_masks, state, noise


def _report_difference(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    difference = (actual.float() - expected.float()).abs()
    print(
        f"{name}: max_abs={difference.max().item():.9g}, "
        f"mean_abs={difference.mean().item():.9g}"
    )


@torch.no_grad()
def test_carrot_pi05_matches_openpi_jax_sampling() -> None:
    device = torch.device("cuda")
    with np.load(Path(OPENPI_GOLDEN), allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata_json"].item())
        assert metadata["schema_version"] == 1
        assert metadata["config"] == "Pi0Config(pi05=True)"
        assert metadata["parameter_dtype"] == "bfloat16"
        assert metadata["image_layout"] == "NIHWC"
        assert metadata["steps"] == [1, 10]

        images, image_masks, tokens, token_masks, state, noise = _load_inputs(
            archive, device
        )
        expected_one_step = torch.from_numpy(archive["openpi_one_step"]).to(device)
        expected_ten_steps = torch.from_numpy(archive["openpi_ten_steps"]).to(device)

    model = PI0Policy.from_pretrained(Path(OPEN_GIGA_CHECKPOINT)).to(device).eval()
    assert model.pi05_enabled
    assert (model.n_action_steps, model.max_action_dim) == (50, 32)

    model.num_steps = 1
    actual_one_step = model.sample_actions(
        images, image_masks, tokens, token_masks, state, noise=noise.clone()
    )
    model.num_steps = 10
    actual_ten_steps = model.sample_actions(
        images, image_masks, tokens, token_masks, state, noise=noise.clone()
    )

    assert actual_one_step.shape == expected_one_step.shape == (1, 50, 32)
    assert actual_ten_steps.shape == expected_ten_steps.shape == (1, 50, 32)
    assert torch.isfinite(actual_one_step).all()
    assert torch.isfinite(actual_ten_steps).all()
    _report_difference("one_step", actual_one_step, expected_one_step)
    _report_difference("ten_steps", actual_ten_steps, expected_ten_steps)
    torch.testing.assert_close(actual_one_step, expected_one_step, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(actual_ten_steps, expected_ten_steps, rtol=1e-2, atol=5e-3)

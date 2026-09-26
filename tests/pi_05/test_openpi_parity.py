from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from carrot.models.pi05.model import PI0Observation, PI0Pytorch

OPENPI_GOLDEN = os.environ.get("CARROT_PI05_OPENPI_GOLDEN")
OPENPI_PYTORCH_GOLDEN = os.environ.get("CARROT_PI05_OPENPI_PYTORCH_GOLDEN")
OPENPI_PYTORCH_CHECKPOINT = os.environ.get("CARROT_PI05_OPENPI_PYTORCH_CHECKPOINT")
type ErrorSummary = tuple[float, float, float, float]
type DifferenceRow = tuple[str, ErrorSummary, ErrorSummary, float]


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


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def _summarize_error(error: torch.Tensor) -> ErrorSummary:
    if not torch.any(error):
        return 0.0, 0.0, 0.0, 0.0
    quantiles = torch.tensor((0.5, 0.9, 0.99), dtype=torch.float64, device=error.device)
    p50, p90, p99 = torch.quantile(error.flatten(), quantiles).tolist()
    return p50, p90, p99, error.max().item()


def _difference_row(name: str, actual: torch.Tensor, expected: torch.Tensor) -> DifferenceRow:
    actual_double = actual.double()
    expected_double = expected.double()
    absolute_error = (actual_double - expected_double).abs()
    relative_error = absolute_error / expected_double.abs().clamp_min(1e-12)
    return (
        name,
        _summarize_error(absolute_error),
        _summarize_error(relative_error),
        calc_diff(actual_double, expected_double).item(),
    )


def _print_difference_table(rows: list[DifferenceRow]) -> None:
    print(
        "| tensor | abs P50 | abs P90 | abs P99 | abs Max | "
        "rel P50 | rel P90 | rel P99 | rel Max | Dice distance |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, absolute, relative, dice_distance in rows:
        values = (*absolute, *relative, dice_distance)
        print(f"| {name} | " + " | ".join(f"{value:.9g}" for value in values) + " |")


@torch.no_grad()
def test_carrot_pi05_matches_openpi_pytorch_sampling() -> None:
    # 验证 Carrot 官方 PyTorch port 与官方 PyTorch golden 的权重及单步采样一致；JAX 仅作诊断参考。
    missing = [
        name
        for name in (
            "CARROT_PI05_OPENPI_GOLDEN",
            "CARROT_PI05_OPENPI_PYTORCH_GOLDEN",
            "CARROT_PI05_OPENPI_PYTORCH_CHECKPOINT",
        )
        if not os.environ.get(name)
    ]
    if missing:
        pytest.fail(f"missing required test assets: {', '.join(missing)}")
    if not torch.cuda.is_available():
        pytest.fail("OpenPI parity requires CUDA")
    device = torch.device("cuda")

    # Arrange：JAX golden 提供固定输入和历史参考值，但不决定测试成败。
    with np.load(Path(OPENPI_GOLDEN), allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata_json"].item())
        assert metadata["schema_version"] == 3
        assert metadata["config"] == "Pi0Config(pi05=True)"
        assert metadata["parameter_dtype"] == "bfloat16"
        assert metadata["image_layout"] == "NIHWC"
        assert metadata["steps"] == [1]

        images, image_masks, tokens, token_masks, state, noise = _load_inputs(archive, device)
        expected_prefix_embeddings = torch.from_numpy(archive["openpi_prefix_embeddings"]).to(
            device
        )
        expected_suffix_embeddings = torch.from_numpy(archive["openpi_suffix_embeddings"]).to(
            device
        )
        expected_adarms_cond = torch.from_numpy(archive["openpi_adarms_cond"]).to(device)
        expected_first_v_t = torch.from_numpy(archive["openpi_first_v_t"]).to(device)
        expected_one_step = torch.from_numpy(archive["openpi_one_step"]).to(device)
        jax_weights = {
            name: torch.from_numpy(archive[f"openpi_{name}"]).to(device)
            for name in (
                "language_embedding_rows",
                "action_in_kernel",
                "action_in_bias",
                "image_patch_kernel",
                "image_patch_bias",
                "image_position_embedding",
                "image_head_kernel",
                "image_head_bias",
            )
        }

    # 官方 PyTorch golden 是严格基准；commit 字段防止基准实现被悄然替换。
    with np.load(Path(OPENPI_PYTORCH_GOLDEN), allow_pickle=False) as archive:
        pytorch_metadata = json.loads(archive["metadata_json"].item())
        assert pytorch_metadata["schema_version"] == 1
        assert pytorch_metadata["openpi_commit"] == "215abfb217dbac7d5f1273282331b9b1866c0479"
        pytorch_prefix_embeddings = torch.from_numpy(
            archive["openpi_pytorch_prefix_embeddings"]
        ).to(device)
        pytorch_suffix_embeddings = torch.from_numpy(
            archive["openpi_pytorch_suffix_embeddings"]
        ).to(device)
        pytorch_adarms_cond = torch.from_numpy(archive["openpi_pytorch_adarms_cond"]).to(device)
        pytorch_first_v_t = torch.from_numpy(archive["openpi_pytorch_first_v_t"]).to(device)
        pytorch_one_step = torch.from_numpy(archive["openpi_pytorch_one_step"]).to(device)
        pytorch_weights = {
            name: torch.from_numpy(archive[f"openpi_pytorch_{name}"]).to(device)
            for name in jax_weights
        }

    # Act：按官方模型层级抽查权重、计算中间量，并用 PI0Observation 执行一步采样。
    model = PI0Pytorch.from_pretrained(Path(OPENPI_PYTORCH_CHECKPOINT)).to(device).eval()
    assert model.config.pi05
    assert (model.config.action_horizon, model.config.action_dim) == (50, 32)
    pi05 = model.paligemma_with_expert
    paligemma = pi05.paligemma.model
    vision_embeddings = paligemma.vision_tower.vision_model.embeddings

    # 选取语言、动作、视觉三条路径的代表性参数，覆盖 checkpoint key 映射。
    actual_weights = {
        "language_embedding_rows": paligemma.language_model.embed_tokens.weight[1:65],
        "action_in_kernel": model.action_in_proj.weight.T,
        "action_in_bias": model.action_in_proj.bias,
        "image_patch_kernel": vision_embeddings.patch_embedding.weight.permute(
            2, 3, 1, 0
        ),
        "image_patch_bias": vision_embeddings.patch_embedding.bias,
        "image_position_embedding": vision_embeddings.position_embedding.weight[None],
        "image_head_kernel": paligemma.multi_modal_projector.linear.weight.T,
        "image_head_bias": paligemma.multi_modal_projector.linear.bias,
    }

    # 同时记录 Carrot/PyTorch/JAX 三组距离，便于区分移植误差和官方实现自身误差。
    difference_rows = []
    for name, actual_weight in actual_weights.items():
        difference_rows.extend(
            [
                _difference_row(f"carrot_vs_jax/weight_{name}", actual_weight, jax_weights[name]),
                _difference_row(
                    f"openpi_pytorch_vs_jax/weight_{name}",
                    pytorch_weights[name],
                    jax_weights[name],
                ),
                _difference_row(
                    f"carrot_vs_openpi_pytorch/weight_{name}",
                    actual_weight,
                    pytorch_weights[name],
                ),
            ]
        )

    # 官方 PyTorch 在 H20 上使用 Flash；显式指定后端，避免版本变化改走 cuDNN。
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        actual_prefix_embeddings, _, _ = model.embed_prefix(
            images, image_masks, tokens, token_masks
        )
    actual_suffix_embeddings, _, _, actual_adarms_cond = model.embed_suffix(
        state, noise, torch.ones((noise.shape[0],), device=device)
    )
    observation = PI0Observation(
        images=dict(
            zip(
                ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
                images,
                strict=True,
            )
        ),
        image_masks=dict(
            zip(
                ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
                image_masks,
                strict=True,
            )
        ),
        state=state,
        tokenized_prompt=tokens,
        tokenized_prompt_mask=token_masks,
    )
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        actual_one_step = model.sample_actions(
            device,
            observation,
            noise=noise.clone(),
            num_steps=1,
        )
    actual_first_v_t = noise - actual_one_step

    assert actual_one_step.shape == pytorch_one_step.shape == (1, 50, 32)
    assert torch.isfinite(actual_one_step).all()

    # prefix 按图像 token 和语言 token 拆开，避免总体误差掩盖单一模态偏差。
    language_length = tokens.shape[1]
    image_prefix_end = expected_prefix_embeddings.shape[1] - language_length
    actual_image_embeddings = actual_prefix_embeddings[:, :image_prefix_end]
    expected_image_embeddings = expected_prefix_embeddings[:, :image_prefix_end]
    actual_language_embeddings = actual_prefix_embeddings[:, image_prefix_end:]
    expected_language_embeddings = expected_prefix_embeddings[:, image_prefix_end:]
    pytorch_image_embeddings = pytorch_prefix_embeddings[:, :image_prefix_end]
    pytorch_language_embeddings = pytorch_prefix_embeddings[:, image_prefix_end:]
    output_tensors = {
        "image_embeddings": (
            actual_image_embeddings,
            pytorch_image_embeddings,
            expected_image_embeddings,
        ),
        "language_embeddings": (
            actual_language_embeddings,
            pytorch_language_embeddings,
            expected_language_embeddings,
        ),
        "suffix_embeddings": (
            actual_suffix_embeddings,
            pytorch_suffix_embeddings,
            expected_suffix_embeddings,
        ),
        "adarms_cond": (actual_adarms_cond, pytorch_adarms_cond, expected_adarms_cond),
        "first_v_t": (actual_first_v_t, pytorch_first_v_t, expected_first_v_t),
        "one_step": (actual_one_step, pytorch_one_step, expected_one_step),
    }

    # 进一步逐相机切分视觉 embedding，诊断相机顺序或单路预处理错误。
    image_tokens_per_camera = image_prefix_end // len(images)
    for camera_index in range(len(images)):
        start = camera_index * image_tokens_per_camera
        end = start + image_tokens_per_camera
        output_tensors[f"image_embeddings_camera_{camera_index}"] = (
            actual_image_embeddings[:, start:end],
            pytorch_image_embeddings[:, start:end],
            expected_image_embeddings[:, start:end],
        )
    for name, (carrot_tensor, pytorch_tensor, jax_tensor) in output_tensors.items():
        difference_rows.extend(
            [
                _difference_row(f"carrot_vs_jax/{name}", carrot_tensor, jax_tensor),
                _difference_row(f"openpi_pytorch_vs_jax/{name}", pytorch_tensor, jax_tensor),
                _difference_row(f"carrot_vs_openpi_pytorch/{name}", carrot_tensor, pytorch_tensor),
            ]
        )
    # JAX 差异只输出诊断表，不作为本测试的通过条件。
    _print_difference_table(difference_rows)
    # Assert：checkpoint 参数必须 BF16 bitwise 一致，权重映射错误不能被容差吞掉。
    for name, actual_weight in actual_weights.items():
        torch.testing.assert_close(
            actual_weight.to(torch.bfloat16),
            pytorch_weights[name].to(torch.bfloat16),
            rtol=0,
            atol=0,
        )
    # NPZ 将部分 BF16 值存为 FP32；1e-2 容忍已定位的 RMSNorm 版本差异，仍逐点比较。
    for name, (carrot_tensor, pytorch_tensor, _) in output_tensors.items():
        torch.testing.assert_close(
            carrot_tensor,
            pytorch_tensor,
            rtol=1e-2,
            atol=1e-2,
            check_dtype=False,
        )
        # Dice 限制整体平方误差，逐点断言负责检出局部偏差。
        dice_distance = calc_diff(carrot_tensor, pytorch_tensor).item()
        assert dice_distance <= 3e-4, (
            f"{name}: FP64 Dice distance {dice_distance:.9g} exceeds 3e-4"
        )

"""Fixed-input official OpenPI PyTorch versus Carrot LIBERO policy parity."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from carrot.models.pi05.inference.policy_config import create_libero_policy

OPENPI_COMMIT = "89d9220c68defd85b85f17acf130e965e58b8369"
OPENPI_SOURCE_SHA256 = "3b5e87f546e2e8effe3dac5b54c42f103bac7dc159c30047f4745ddaf5e5264d"
TOKENIZER_SHA256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
ACTION_ATOL = 6e-3
GOLDEN = os.environ.get("CARROT_PI05_LIBERO_GOLDEN")
CHECKPOINT = os.environ.get("CARROT_PI05_LIBERO_CHECKPOINT")
TOKENIZER = os.environ.get("CARROT_PI05_LIBERO_TOKENIZER")
pytestmark = pytest.mark.skipif(
    not GOLDEN or not CHECKPOINT or not TOKENIZER or not torch.cuda.is_available(),
    reason="set LIBERO golden, checkpoint, tokenizer, and use a CUDA host",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _error_quantiles(values: np.ndarray) -> tuple[float, float, float, float]:
    return tuple(float(value) for value in (*np.quantile(values, (0.5, 0.9, 0.99)), values.max()))


def _compare_float(
    name: str, actual: np.ndarray, expected: np.ndarray, *, atol: float = 1e-3
) -> None:
    assert actual.shape == expected.shape, f"{name}: {actual.shape} != {expected.shape}"
    actual64 = actual.astype(np.float64)
    expected64 = expected.astype(np.float64)
    assert np.isfinite(actual64).all() and np.isfinite(expected64).all(), name
    absolute = np.abs(actual64 - expected64)
    relative = absolute / np.maximum(np.abs(expected64), 1e-12)
    denominator = np.square(actual64).sum() + np.square(expected64).sum()
    dot_product = np.multiply(actual64, expected64).sum()
    dice = 0.0 if denominator == 0 else 1 - 2 * dot_product / denominator
    values = (*_error_quantiles(absolute), *_error_quantiles(relative), dice)
    print(f"| {name} | " + " | ".join(f"{value:.9g}" for value in values) + " |", flush=True)
    np.testing.assert_allclose(actual64, expected64, rtol=1e-3, atol=atol, err_msg=name)


def _compare_exact(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_array_equal(actual, expected, err_msg=name)
    print(f"{name}: exact", flush=True)


@torch.no_grad()
def test_libero_policy_matches_openpi_pytorch() -> None:
    # 用真实 LIBERO 样本和官方 checkpoint 锁定完整训前推理链；首次失败字段定位偏差来源。
    checkpoint = Path(CHECKPOINT)
    stats_path = checkpoint / "assets/physical-intelligence/libero/norm_stats.json"
    with np.load(Path(GOLDEN), allow_pickle=False) as golden:
        # 先核验参考版本、模型配置和 stats，防止不同资产恰好产生近似动作。
        metadata = json.loads(golden["metadata_json"].item())
        assert metadata["schema_version"] == 1
        assert metadata["openpi_commit"] == OPENPI_COMMIT
        assert metadata["openpi_source_sha256"] == OPENPI_SOURCE_SHA256
        assert metadata["config"] == "pi05_libero"
        assert metadata["checkpoint"] == str(checkpoint.resolve())
        assert metadata["checkpoint_size"] == (checkpoint / "model.safetensors").stat().st_size
        assert metadata["checkpoint_config_sha256"] == _sha256(checkpoint / "config.json")
        assert metadata["norm_stats_sha256"] == _sha256(stats_path)
        assert metadata["tokenizer_sha256"] == TOKENIZER_SHA256
        assert metadata["tokenizer_sha256"] == _sha256(Path(TOKENIZER) / "tokenizer.model")
        assert metadata["num_steps"] == 10
        assert metadata["pytorch_compile_mode"] == "max-autotune"

        # 同一份原始图像、state、task prompt 与固定噪声进入 Carrot 的实际 policy 入口。
        raw = {
            "observation/state": golden["raw_state"].copy(),
            "observation/image": golden["raw_base_image"].copy(),
            "observation/wrist_image": golden["raw_wrist_image"].copy(),
            "prompt": golden["raw_prompt"].item(),
        }
        noise = golden["noise"].copy()
        policy = create_libero_policy(
            checkpoint, device="cuda", tokenizer_path=Path(TOKENIZER), num_steps=10
        )
        assert policy._model.config.action_horizon == 10
        assert policy._model.config.action_dim == 32
        transformed = policy._input_transform(dict(raw))
        observation = policy._to_observation(transformed)

        # 离散字段必须精确一致；浮点字段先打印 FP64 分布，再执行逐点断言。
        print(
            "| tensor | abs P50 | abs P90 | abs P99 | abs Max | "
            "rel P50 | rel P90 | rel P99 | rel Max | Dice distance |",
            flush=True,
        )
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|", flush=True)
        for key in IMAGE_KEYS:
            _compare_float(
                f"image/{key}",
                observation.images[key][0].float().cpu().numpy(),
                golden[f"image_{key}"],
            )
            _compare_exact(
                f"image_mask/{key}",
                observation.image_masks[key][0].cpu().numpy(),
                golden[f"image_mask_{key}"],
            )
        _compare_float(
            "normalized_state", observation.state[0].float().cpu().numpy(), golden["state"]
        )
        _compare_exact("tokens", observation.tokenized_prompt[0].cpu().numpy(), golden["tokens"])
        _compare_exact(
            "token_mask", observation.tokenized_prompt_mask[0].cpu().numpy(), golden["token_mask"]
        )

        # 模型输出用相同 noise 和采样步数；最终调用公开 infer 以覆盖动作解码。
        raw_actions = policy._model.sample_actions(
            torch.device("cuda"),
            observation,
            noise=torch.from_numpy(noise[None]).to("cuda"),
            num_steps=10,
        )[0]
        # 两端 Torch/Transformers 版本不同，BF16 误差在 10 步 Euler 更新中累积。
        _compare_float(
            "raw_actions_32d",
            raw_actions.float().cpu().numpy(),
            golden["raw_actions"],
            atol=ACTION_ATOL,
        )
        decoded = policy.infer(raw, noise=noise)["actions"]
        _compare_float("decoded_actions_7d", decoded, golden["decoded_actions"], atol=ACTION_ATOL)

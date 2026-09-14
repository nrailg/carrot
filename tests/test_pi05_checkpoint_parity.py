from __future__ import annotations

import json
import os
from contextlib import ExitStack
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from tests.pi05_checkpoint_utils import lerobot_key_for_open_giga_key

# 两个变量都是 Hugging Face checkpoint 的本地 snapshot 根目录，不是权重文件路径：
# LeRobot 目录内应有 model.safetensors，OpenGiga 目录内应有 Diffusers index 和分片权重。
# 本次 Gemini 验证实际指向：
#   /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/lerobot/pi05_base
#   /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/open-gigaai/pi05_base
# 大权重只在显式配置这两个目录的临时验证环境中读取，普通单测默认跳过。
LEROBOT_CHECKPOINT = os.environ.get("CARROT_PI05_LEROBOT_CHECKPOINT")
OPEN_GIGA_CHECKPOINT = os.environ.get("CARROT_PI05_OPEN_GIGA_CHECKPOINT")
pytestmark = pytest.mark.skipif(
    LEROBOT_CHECKPOINT is None or OPEN_GIGA_CHECKPOINT is None,
    reason="set CARROT_PI05_LEROBOT_CHECKPOINT and CARROT_PI05_OPEN_GIGA_CHECKPOINT",
)


@pytest.mark.parametrize(
    ("open_giga_key", "lerobot_key"),
    [
        (
            "paligemma_with_expert.layers.2.self_attn.q_proj.0.weight",
            "paligemma_with_expert.paligemma.model.language_model.layers.2.self_attn.q_proj.weight",
        ),
        (
            "paligemma_with_expert.layers.3.mlps.1.down_proj.weight",
            "paligemma_with_expert.gemma_expert.model.layers.3.mlp.down_proj.weight",
        ),
    ],
)
def test_open_giga_to_lerobot_key_mapping(open_giga_key: str, lerobot_key: str) -> None:
    assert lerobot_key_for_open_giga_key(open_giga_key) == lerobot_key


def _keeps_fp32_value(open_giga_key: str) -> bool:
    if not open_giga_key.startswith("paligemma_with_expert."):
        return True
    return (
        open_giga_key.startswith("paligemma_with_expert.norms.")
        or open_giga_key.startswith("paligemma_with_expert.vision_tower.embeddings.")
        or ".input_layernorms." in open_giga_key
        or ".post_attention_layernorms." in open_giga_key
    )


def test_lerobot_and_open_giga_base_weights_match_mixed_precision_values() -> None:
    lerobot_path = Path(LEROBOT_CHECKPOINT)
    open_giga_path = Path(OPEN_GIGA_CHECKPOINT)
    with (open_giga_path / "diffusion_pytorch_model.safetensors.index.json").open() as stream:
        weight_map = json.load(stream)["weight_map"]

    mismatches: list[str] = []
    mapped_lerobot_keys: set[str] = set()
    fp32_tensors = 0
    bf16_tensors = 0
    with ExitStack() as stack:
        lerobot = stack.enter_context(
            safe_open(lerobot_path / "model.safetensors", framework="pt", device="cpu")
        )
        shards = {
            shard: stack.enter_context(
                safe_open(open_giga_path / shard, framework="pt", device="cpu")
            )
            for shard in set(weight_map.values())
        }
        lerobot_keys = set(lerobot.keys())

        for open_giga_key, shard in weight_map.items():
            lerobot_key = lerobot_key_for_open_giga_key(open_giga_key)
            mapped_lerobot_keys.add(lerobot_key)
            if lerobot_key not in lerobot_keys:
                mismatches.append(f"missing LeRobot key: {lerobot_key}")
                continue

            actual = lerobot.get_tensor(lerobot_key)
            expected = shards[shard].get_tensor(open_giga_key)
            if actual.dtype != torch.float32 or expected.dtype != torch.float32:
                mismatches.append(
                    f"dtype {open_giga_key}: LeRobot {actual.dtype} != OpenGiga {expected.dtype}"
                )
                continue
            if actual.shape != expected.shape:
                shape_difference = (
                    f"LeRobot {tuple(actual.shape)} != OpenGiga {tuple(expected.shape)}"
                )
                mismatches.append(f"shape {open_giga_key}: {shape_difference}")
                continue
            if _keeps_fp32_value(open_giga_key):
                fp32_tensors += 1
                expected_value = expected
                precision = "FP32"
            else:
                bf16_tensors += 1
                expected_value = expected.to(torch.bfloat16).to(torch.float32)
                precision = "BF16 round-trip"
            if not torch.equal(actual, expected_value):
                max_abs = (actual - expected_value).abs().max().item()
                mismatches.append(
                    f"value {open_giga_key} ({precision}): max_abs={max_abs:.9g}"
                )

    unused_lerobot_keys = lerobot_keys - mapped_lerobot_keys
    assert unused_lerobot_keys == {"paligemma_with_expert.gemma_expert.lm_head.weight"}
    assert (fp32_tensors, bf16_tensors) == (122, 689)
    assert not mismatches, "\n".join(mismatches[:50])

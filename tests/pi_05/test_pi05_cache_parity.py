from typing import Any

import pytest
import torch

from carrot.models.pi05.model.gemma_pytorch import PaliGemmaWithExpertModel
from carrot.models.pi05.model.pi0_pytorch import _get_gemma_config, make_att_2d_masks
from carrot.models.pi05.model.transformers_replace.models.gemma.configuration_gemma import (
    GemmaConfig,
)
from carrot.models.pi05.model.transformers_replace.models.gemma.modeling_gemma import GemmaAttention


@pytest.fixture(scope="module")
def model() -> PaliGemmaWithExpertModel:
    # 用真实四层计算路径，随机参数和 adaptive norm 保留两分支的层间依赖。
    torch.manual_seed(42)
    config = _get_gemma_config("dummy")
    return PaliGemmaWithExpertModel(
        config, config, use_adarms=[False, True], precision="float32"
    ).cuda().eval()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_joint_and_cached_condition_match(
    model: PaliGemmaWithExpertModel, dtype: torch.dtype
) -> None:
    # 同 backend 下拆分计算应保留有效 token 的结果，suffix 不得写入固定条件缓存。
    # 固定输入含被屏蔽的 prefix token，排除仅在无 padding 情况成立的伪一致性。
    model.to(dtype=dtype)
    model.paligemma.language_model.config._attn_implementation = "eager"
    model.gemma_expert.model.config._attn_implementation = "eager"
    torch.manual_seed(17)
    prefix = torch.randn(1, 12, 64, device="cuda", dtype=dtype)
    suffix = torch.randn(1, 10, 64, device="cuda", dtype=dtype)
    condition = torch.randn(1, 64, device="cuda", dtype=dtype)
    prefix_pad = torch.tensor([[True] * 8 + [False] * 4], device="cuda")
    suffix_pad = torch.ones(1, 10, device="cuda", dtype=torch.bool)
    joint_pad = torch.cat((prefix_pad, suffix_pad), dim=1)
    markers = torch.tensor([[False] * 12 + [True] + [False] * 9], device="cuda")
    allowed = make_att_2d_masks(joint_pad, markers)
    mask = torch.where(allowed[:, None], 0.0, -2.3819763e38)
    positions = torch.cumsum(joint_pad, dim=1) - 1

    # 完整路径和拆分路径用同一 mask 子块及 position，避免测试引入额外变量。
    (joint_prefix, joint_suffix), _ = model.forward(
        inputs_embeds=[prefix, suffix], attention_mask=mask, position_ids=positions,
        past_key_values=None, use_cache=False, adarms_cond=[None, condition],
    )
    (cached_prefix, _), cache = model.forward(
        inputs_embeds=[prefix, None], attention_mask=mask[:, :, :12, :12],
        position_ids=positions[:, :12], past_key_values=None, use_cache=True,
    )
    saved: list[tuple[Any, Any]] = [(k.clone(), v.clone()) for k, v in cache]
    (_, cached_suffix), _ = model.forward(
        inputs_embeds=[None, suffix], attention_mask=mask[:, :, 12:, :],
        position_ids=positions[:, 12:], past_key_values=cache, use_cache=False,
        adarms_cond=[None, condition],
    )

    # BF16 允许低精度算子布局造成的小偏差，FP32 使用更严格阈值；不比较全 masked 查询。
    tolerance = 2e-4 if dtype == torch.float32 else 0.04
    torch.testing.assert_close(joint_prefix[:, :8], cached_prefix[:, :8], atol=tolerance, rtol=0)
    torch.testing.assert_close(joint_suffix, cached_suffix, atol=tolerance, rtol=0)
    for (key, value), (saved_key, saved_value) in zip(cache, saved, strict=True):
        torch.testing.assert_close(key, saved_key, atol=0, rtol=0)
        torch.testing.assert_close(value, saved_value, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("boolean_mask", [False, True])
@torch.no_grad()
def test_sdpa_respects_block_mask(dtype: torch.dtype, boolean_mask: bool) -> None:
    # 合法 dtype 的 mask 必须屏蔽后续 block；错误浮点 dtype 必须明确拒绝，bool 语义保持。
    # 构造条件/动作两个双向 block，条件 query 禁止读动作；改变动作应不影响条件输出。
    torch.manual_seed(88)
    config = GemmaConfig(
        hidden_size=64, intermediate_size=128, num_attention_heads=8,
        num_key_value_heads=1, head_dim=256, attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    attention = GemmaAttention(config, layer_idx=0).cuda().to(dtype).eval()
    hidden = torch.randn(1, 64, 64, device="cuda", dtype=dtype)
    altered = hidden.clone()
    altered[:, 32:] += 10
    valid = torch.ones(1, 64, device="cuda", dtype=torch.bool)
    markers = torch.tensor([[False] * 32 + [True] + [False] * 31], device="cuda")
    allowed = make_att_2d_masks(valid, markers)[:, None]
    mask = allowed if boolean_mask else torch.where(allowed, 0.0, -2.3819763e38).to(dtype)
    cos = torch.ones(1, 64, 256, device="cuda", dtype=dtype)
    sin = torch.zeros_like(cos)

    # 使用真实 GemmaAttention 的 backend 适配接口；调用方负责正确构造 mask dtype。
    original_output, _ = attention(hidden, (cos, sin), attention_mask=mask)
    altered_output, _ = attention(altered, (cos, sin), attention_mask=mask)

    # 前 block 的合法 K/V 未改动，应精确相同；大偏差意味着 mask 未正确生效。
    torch.testing.assert_close(original_output[:, :32], altered_output[:, :32], atol=0, rtol=0)

    # 反向精度组合也必须在调用 SDPA 前失败，避免底层静默转换掩盖调用方错误。
    if not boolean_mask:
        wrong_dtype = torch.float32 if dtype == torch.bfloat16 else torch.bfloat16
        with pytest.raises(AssertionError, match="SDPA mask dtype"):
            attention(hidden, (cos, sin), attention_mask=mask.to(wrong_dtype))

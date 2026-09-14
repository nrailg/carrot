from __future__ import annotations

import re

_BRANCHED_LAYER_KEY = re.compile(
    r"^paligemma_with_expert\.layers\.(?P<layer>\d+)\."
    r"(?P<module>input_layernorms|post_attention_layernorms|mlps)\."
    r"(?P<branch>[01])\.(?P<suffix>.+)$"
)
_ATTENTION_KEY = re.compile(
    r"^paligemma_with_expert\.layers\.(?P<layer>\d+)\.self_attn\."
    r"(?P<projection>[qkvo]_proj)\.(?P<branch>[01])\.(?P<suffix>.+)$"
)


def lerobot_key_for_open_giga_key(key: str) -> str:
    if not key.startswith("paligemma_with_expert."):
        return key

    if key == "paligemma_with_expert.embed_tokens.weight":
        return "paligemma_with_expert.paligemma.lm_head.weight"
    if key == "paligemma_with_expert.norms.0.weight":
        return "paligemma_with_expert.paligemma.model.language_model.norm.weight"
    if key.startswith("paligemma_with_expert.norms.1."):
        suffix = key.removeprefix("paligemma_with_expert.norms.1.")
        return f"paligemma_with_expert.gemma_expert.model.norm.{suffix}"
    if key.startswith("paligemma_with_expert.vision_tower."):
        suffix = key.removeprefix("paligemma_with_expert.vision_tower.")
        return f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.{suffix}"
    if key.startswith("paligemma_with_expert.multi_modal_projector."):
        suffix = key.removeprefix("paligemma_with_expert.multi_modal_projector.")
        return f"paligemma_with_expert.paligemma.model.multi_modal_projector.{suffix}"

    attention_match = _ATTENTION_KEY.fullmatch(key)
    match = attention_match or _BRANCHED_LAYER_KEY.fullmatch(key)
    if match is None:
        raise KeyError(f"No LeRobot mapping for OpenGiga key: {key}")

    branch = match.group("branch")
    model_prefix = (
        "paligemma_with_expert.paligemma.model.language_model"
        if branch == "0"
        else "paligemma_with_expert.gemma_expert.model"
    )
    if attention_match is not None:
        suffix = f"self_attn.{match.group('projection')}.{match.group('suffix')}"
    else:
        module = {
            "input_layernorms": "input_layernorm",
            "post_attention_layernorms": "post_attention_layernorm",
            "mlps": "mlp",
        }[match.group("module")]
        suffix = f"{module}.{match.group('suffix')}"
    return f"{model_prefix}.layers.{match.group('layer')}.{suffix}"

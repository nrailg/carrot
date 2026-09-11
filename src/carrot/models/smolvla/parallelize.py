"""FSDP2 wrapping plan for LeRobot SmolVLA."""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn

from carrot.modeling.config import FSDPConfig
from carrot.modeling.parallelizer import ModelParallelizer


def _module(parent: object, name: str) -> nn.Module:
    value = getattr(parent, name, None)
    if not isinstance(value, nn.Module):
        raise TypeError(f"SmolVLA structure requires module {name!r} on {type(parent).__name__}")
    return value


class SmolVLAParallelizer(ModelParallelizer):
    """Declare only modules reached through ``nn.Module.__call__`` by SmolVLA."""

    def validate_config(self, config: FSDPConfig) -> None:
        if config.forward_prefetch or config.backward_prefetch:
            raise ValueError("SmolVLA does not support linear FSDP prefetch ordering")

    def fsdp_units(self, model: nn.Module) -> Sequence[nn.Module]:
        flow_model = getattr(model, "model", None)
        if not isinstance(flow_model, nn.Module):
            raise TypeError("SmolVLA policy must expose its flow-matching module as 'model'")
        vlm_with_expert = _module(flow_model, "vlm_with_expert")
        get_vlm_model = getattr(vlm_with_expert, "get_vlm_model", None)
        if not callable(get_vlm_model):
            raise TypeError("SmolVLA VLM/expert module must expose get_vlm_model()")
        vlm_model = get_vlm_model()
        text_model = _module(vlm_model, "text_model")
        expert_model = _module(vlm_with_expert, "lm_expert")

        units = [
            _module(vlm_model, "vision_model"),
            _module(vlm_model, "connector"),
            _module(text_model, "embed_tokens"),
        ]
        get_model_layers = getattr(vlm_with_expert, "get_model_layers", None)
        if not callable(get_model_layers):
            raise TypeError("SmolVLA VLM/expert module must expose get_model_layers()")
        layer_groups = get_model_layers([text_model, expert_model])
        for layers in layer_groups:
            for layer in layers:
                if layer is None:
                    continue
                attention = _module(layer, "self_attn")
                units.extend(
                    [
                        _module(layer, "input_layernorm"),
                        _module(attention, "q_proj"),
                        _module(attention, "k_proj"),
                        _module(attention, "v_proj"),
                        _module(attention, "o_proj"),
                        _module(layer, "post_attention_layernorm"),
                        _module(layer, "mlp"),
                    ]
                )
        units.extend(
            [
                _module(text_model, "norm"),
                _module(expert_model, "norm"),
                _module(flow_model, "state_proj"),
                _module(flow_model, "action_in_proj"),
                _module(flow_model, "action_out_proj"),
                _module(flow_model, "action_time_mlp_in"),
                _module(flow_model, "action_time_mlp_out"),
            ]
        )

        unique_units: list[nn.Module] = []
        seen = set()
        for unit in units:
            if id(unit) not in seen:
                seen.add(id(unit))
                unique_units.append(unit)
        return tuple(unique_units)

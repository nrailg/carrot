"""FSDP2 wrapping plan for LeRobot SmolVLA."""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn

from carrot.modeling.config import FSDPConfig
from carrot.modeling.parallelizer import ModelParallelizer


class SmolVLAParallelizer(ModelParallelizer):
    """Declare only modules reached through ``nn.Module.__call__`` by SmolVLA."""

    def validate_config(self, config: FSDPConfig) -> None:
        if config.forward_prefetch or config.backward_prefetch:
            raise ValueError("SmolVLA does not support linear FSDP prefetch ordering")

    def fsdp_units(self, model: nn.Module) -> Sequence[nn.Module]:
        try:
            flow_model = model.model
        except AttributeError as error:
            raise TypeError(
                "SmolVLA policy must expose its flow-matching module as 'model'"
            ) from error
        vlm_with_expert = flow_model.vlm_with_expert
        vlm_model = vlm_with_expert.get_vlm_model()
        text_model = vlm_model.text_model
        expert_model = vlm_with_expert.lm_expert

        units = [
            vlm_model.vision_model,
            vlm_model.connector,
            text_model.embed_tokens,
        ]
        layer_groups = vlm_with_expert.get_model_layers([text_model, expert_model])
        for layers in layer_groups:
            for layer in layers:
                if layer is None:
                    continue
                attention = layer.self_attn
                units.extend(
                    [
                        attention.q_proj,
                        attention.k_proj,
                        attention.v_proj,
                        attention.o_proj,
                        layer.mlp,
                    ]
                )
        units.extend(
            [
                flow_model.state_proj,
                flow_model.action_in_proj,
                flow_model.action_out_proj,
                flow_model.action_time_mlp_in,
                flow_model.action_time_mlp_out,
            ]
        )

        unique_units: list[nn.Module] = []
        seen = set()
        for unit in units:
            if id(unit) not in seen:
                seen.add(id(unit))
                unique_units.append(unit)
        return tuple(unique_units)

"""FSDP2 wrapping plan for the native PI0.5 policy."""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn

from carrot.modeling.parallelizer import ModelParallelizer


class Pi05Parallelizer(ModelParallelizer):
    """Wrap only modules reached through PI0.5's layer ``__call__`` path."""

    def fsdp_units(self, model: nn.Module) -> Sequence[nn.Module]:
        backbone = model.paligemma_with_expert
        vision_layers = backbone.vision_tower.encoder.layers
        decoder_layers = backbone.layers
        # TODO 确认下 vision 是不是 AMP 的
        return tuple([*vision_layers, *decoder_layers])

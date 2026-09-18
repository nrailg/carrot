"""FSDP2 wrapping plan for the native PI0.5 model."""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn

from carrot.parallel.parallelizer import ModelParallelizer


class Pi05Parallelizer(ModelParallelizer):
    """Wrap only modules reached through PI0.5's layer ``__call__`` path."""

    def fsdp_units(self, model: nn.Module) -> Sequence[nn.Module]:
        backbone = model.paligemma_with_expert
        # TODO: Expose callable paired decoder blocks so FSDP can reshard them per layer.
        return tuple(backbone.paligemma.model.vision_tower.vision_model.encoder.layers)

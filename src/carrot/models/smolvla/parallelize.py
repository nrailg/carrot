"""FSDP2 wrapping plan for LeRobot SmolVLA."""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn

from carrot.modeling.parallelizer import ModelParallelizer


class SmolVLAParallelizer(ModelParallelizer):
    """Shard the flow-matching model at a forward boundary used by LeRobot."""

    def fsdp_units(self, model: nn.Module) -> Sequence[nn.Module]:
        flow_model = getattr(model, "model", None)
        if not isinstance(flow_model, nn.Module):
            raise TypeError("SmolVLA policy must expose its flow-matching module as 'model'")
        return (flow_model,)

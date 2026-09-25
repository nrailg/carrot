"""FSDP2 model parallelization independent of a training loop."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

from carrot.parallel.config import FSDPConfig


class ModelParallelizer(ABC):
    """Declare FSDP units in their forward execution order."""

    @abstractmethod
    def fsdp_units(self, model: nn.Module) -> Sequence[nn.Module]:
        """Return non-root modules whose forward is invoked by the model."""

    def validate_config(self, config: FSDPConfig) -> None:
        del config


def _dtype(name: str) -> torch.dtype:
    dtypes = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    assert name in dtypes, f"unsupported dtype {name!r}"
    return dtypes[name]


def _set_prefetch(units: Sequence[nn.Module], forward: int, backward: int) -> None:
    if forward:
        for index, unit in enumerate(units):
            following = units[index + 1 : index + 1 + forward]
            if following:
                unit.set_modules_to_forward_prefetch(list(following))
    if backward:
        for index, unit in enumerate(units):
            preceding = units[max(0, index - backward) : index]
            if preceding:
                unit.set_modules_to_backward_prefetch(list(reversed(preceding)))


def parallelize_model(
    model: nn.Module,
    parallelizer: ModelParallelizer,
    config: FSDPConfig,
    *,
    mesh: DeviceMesh | None = None,
) -> nn.Module:
    """Apply bottom-up composable FSDP2 wrapping and return the same model."""
    if not config.enabled:
        return model
    assert dist.is_initialized(), "torch.distributed must be initialized before applying FSDP2"

    # FSDP mixed precision casts full parameters only for forward/backward. Keep
    # the sharded master parameters in FP32 so AdamW moments are FP32 as well.
    model.to(dtype=torch.float32)
    for name, parameter in model.named_parameters():
        assert not parameter.dtype.is_floating_point or parameter.dtype is torch.float32, (
            f"FSDP requires FP32 master weights, got {name}={parameter.dtype}"
        )

    policy = MixedPrecisionPolicy(
        param_dtype=_dtype(config.param_dtype),
        reduce_dtype=_dtype(config.reduce_dtype),
        cast_forward_inputs=True,
    )
    parallelizer.validate_config(config)
    units = tuple(parallelizer.fsdp_units(model))
    assert units, "FSDP2 requires at least one non-root forward unit"
    assert all(unit is not model for unit in units), "fsdp_units must not contain the root model"
    assert len({id(unit) for unit in units}) == len(units), "fsdp_units must be unique"
    descendants = {id(module) for module in model.modules() if module is not model}
    assert all(id(unit) in descendants for unit in units), (
        "every FSDP unit must be a descendant of the root model"
    )

    for unit in units:
        assert not isinstance(unit, FSDPModule), (
            "model contains an FSDP unit before parallelization"
        )
        fully_shard(
            unit,
            mesh=mesh,
            mp_policy=policy,
            reshard_after_forward=config.reshard_after_forward,
        )

    fully_shard(
        model,
        mesh=mesh,
        mp_policy=policy,
        reshard_after_forward=config.reshard_after_forward,
    )
    _set_prefetch(units, config.forward_prefetch, config.backward_prefetch)
    return model

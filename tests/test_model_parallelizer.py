import pytest
import torch
from torch import nn

from carrot.models.pi05.parallelize import Pi05Parallelizer
from carrot.parallel import FSDPConfig, ModelParallelizer, parallelize_model


class FakePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Linear(2, 2)

    def forward(self, inputs):
        return self.model(inputs)


class EmptyParallelizer(ModelParallelizer):
    def fsdp_units(self, model: nn.Module):
        return ()


class SingleUnitParallelizer(ModelParallelizer):
    def fsdp_units(self, model: nn.Module):
        return (model.model,)


def test_disabled_fsdp_preserves_model() -> None:
    model = FakePolicy()

    result = parallelize_model(model, SingleUnitParallelizer(), FSDPConfig(enabled=False))

    assert result is model


def test_fsdp_requires_initialized_process_group() -> None:
    with pytest.raises(RuntimeError, match="torch.distributed"):
        parallelize_model(FakePolicy(), EmptyParallelizer(), FSDPConfig())


def test_fsdp_uses_fp32_master_and_configured_mixed_precision(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr("carrot.parallel.parallelizer.dist.is_initialized", lambda: True)
    monkeypatch.setattr(
        "carrot.parallel.parallelizer.fully_shard",
        lambda module, **kwargs: calls.append((module, kwargs)),
    )
    model = FakePolicy().to(dtype=torch.bfloat16)

    parallelize_model(
        model,
        SingleUnitParallelizer(),
        FSDPConfig(param_dtype="bfloat16", reduce_dtype="float32"),
    )

    assert len(calls) == 2
    assert all(parameter.dtype is torch.float32 for parameter in model.parameters())
    for _, kwargs in calls:
        assert kwargs["mp_policy"].param_dtype is torch.bfloat16
        assert kwargs["mp_policy"].reduce_dtype is torch.float32
        assert kwargs["mp_policy"].cast_forward_inputs is True


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_tower = nn.Module()
        self.vision_tower.encoder = nn.Module()
        self.vision_tower.encoder.layers = nn.ModuleList([nn.Linear(2, 2)])
        self.layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])


class _NativePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.paligemma_with_expert = _Backbone()


class _SFTPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.policy = _NativePolicy()


def test_pi05_parallelizer_wraps_transformer_layers() -> None:
    model = _SFTPolicy()

    units = Pi05Parallelizer().fsdp_units(model)

    assert units == tuple(
        [
            *model.policy.paligemma_with_expert.vision_tower.encoder.layers,
            *model.policy.paligemma_with_expert.layers,
        ]
    )

import pytest
from torch import nn

from carrot.modeling import FSDPConfig, ModelParallelizer, parallelize_model
from carrot.models.smolvla import SmolVLAParallelizer


class FakePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Linear(2, 2)

    def forward(self, inputs):
        return self.model(inputs)


class EmptyParallelizer(ModelParallelizer):
    def fsdp_units(self, model: nn.Module):
        return ()


def test_disabled_fsdp_preserves_model() -> None:
    model = FakePolicy()

    result = parallelize_model(model, SmolVLAParallelizer(), FSDPConfig(enabled=False))

    assert result is model


def test_fsdp_requires_initialized_process_group() -> None:
    with pytest.raises(RuntimeError, match="torch.distributed"):
        parallelize_model(FakePolicy(), EmptyParallelizer(), FSDPConfig())


def test_smolvla_parallelizer_requires_flow_model() -> None:
    with pytest.raises(TypeError, match="flow-matching"):
        SmolVLAParallelizer().fsdp_units(nn.Linear(2, 2))

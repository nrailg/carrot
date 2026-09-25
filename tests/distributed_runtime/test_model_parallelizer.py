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
    # 验证关闭 FSDP 时 parallelize_model 不替换模型对象。
    # Arrange：构造一个带可包装子模块的最小 policy。
    model = FakePolicy()

    # Act：通过 disabled 配置调用统一 parallelize 入口。
    result = parallelize_model(model, SingleUnitParallelizer(), FSDPConfig(enabled=False))

    # Assert：关闭路径必须原样返回调用方持有的模型。
    assert result is model


def test_fsdp_requires_initialized_process_group() -> None:
    # 验证启用 FSDP 前必须已有 torch.distributed process group。
    # Act/Assert：未初始化 distributed 时，入口应在访问 wrapping 单元前明确失败。
    with pytest.raises(AssertionError, match="torch.distributed"):
        parallelize_model(FakePolicy(), EmptyParallelizer(), FSDPConfig())


def test_fsdp_uses_fp32_master_and_configured_mixed_precision(monkeypatch) -> None:
    # 验证 FSDP wrapping 使用 FP32 master 参数并传递配置的混合精度策略。
    calls = []

    # Arrange：拦截 fully_shard，隔离真实 distributed/GPU，只观察 parallelizer 的调用契约。
    monkeypatch.setattr("carrot.parallel.parallelizer.dist.is_initialized", lambda: True)
    monkeypatch.setattr(
        "carrot.parallel.parallelizer.fully_shard",
        lambda module, **kwargs: calls.append((module, kwargs)),
    )
    model = FakePolicy().to(dtype=torch.bfloat16)

    # Act：用 fake fully_shard 记录每次 wrapping 的模块和 mixed-precision 配置。
    parallelize_model(
        model,
        SingleUnitParallelizer(),
        FSDPConfig(param_dtype="bfloat16", reduce_dtype="float32"),
    )

    # Assert：模型参数保持 FP32，且每个 wrapping 单元收到预期策略。
    assert len(calls) == 2
    assert all(parameter.dtype is torch.float32 for parameter in model.parameters())
    for _, kwargs in calls:
        assert kwargs["mp_policy"].param_dtype is torch.bfloat16
        assert kwargs["mp_policy"].reduce_dtype is torch.float32
        assert kwargs["mp_policy"].cast_forward_inputs is True


class _Backbone(nn.Module):
    # 只构造 fsdp_units 会访问的官方层级，避免测试绑定无关模型实现。
    def __init__(self) -> None:
        super().__init__()
        self.paligemma = nn.Module()
        self.paligemma.model = nn.Module()
        self.paligemma.model.vision_tower = nn.Module()
        self.paligemma.model.vision_tower.vision_model = nn.Module()
        self.paligemma.model.vision_tower.vision_model.encoder = nn.Module()
        self.paligemma.model.vision_tower.vision_model.encoder.layers = nn.ModuleList(
            [nn.Linear(2, 2)]
        )


class _NativePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.paligemma_with_expert = _Backbone()


def test_pi05_parallelizer_wraps_transformer_layers() -> None:
    # 验证官方 PI0.5 层级下仅把 vision encoder layers 作为 FSDP 单元。
    model = _NativePolicy()

    # Act：解析官方 paligemma.model.vision_tower.vision_model 层级。
    units = Pi05Parallelizer().fsdp_units(model)

    # Assert：返回的顺序和模块实例与 vision encoder 完全一致。
    assert units == tuple(
        model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers
    )

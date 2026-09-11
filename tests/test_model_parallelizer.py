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


class FakeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(2, 2)
        self.k_proj = nn.Linear(2, 2)
        self.v_proj = nn.Linear(2, 2)
        self.o_proj = nn.Linear(2, 2)


class FakeDecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(2)
        self.self_attn = FakeAttention()
        self.post_attention_layernorm = nn.LayerNorm(2)
        self.mlp = nn.Linear(2, 2)


class FakeTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(2, 2)
        self.layers = nn.ModuleList([FakeDecoderLayer()])
        self.norm = nn.LayerNorm(2)


class FakeVLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_model = nn.Linear(2, 2)
        self.connector = nn.Linear(2, 2)
        self.text_model = FakeTextModel()


class FakeVLMWithExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vlm = FakeVLM()
        self.lm_expert = FakeTextModel()

    def get_vlm_model(self):
        return self.vlm

    @staticmethod
    def get_model_layers(models):
        return [model.layers for model in models]


class FakeFlowModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vlm_with_expert = FakeVLMWithExpert()
        for name in (
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ):
            setattr(self, name, nn.Linear(2, 2))


class StructuredFakePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeFlowModel()


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


def test_smolvla_parallelizer_selects_hook_safe_leaf_modules() -> None:
    policy = StructuredFakePolicy()

    units = SmolVLAParallelizer().fsdp_units(policy)

    assert policy.model not in units
    assert policy.model.vlm_with_expert.vlm.vision_model in units
    assert policy.model.vlm_with_expert.vlm.text_model.layers[0].self_attn.q_proj in units
    assert policy.model.action_out_proj in units
    assert len({id(unit) for unit in units}) == len(units)


def test_smolvla_parallelizer_rejects_linear_prefetch() -> None:
    with pytest.raises(ValueError, match="prefetch"):
        SmolVLAParallelizer().validate_config(FSDPConfig(forward_prefetch=1))

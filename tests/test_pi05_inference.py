import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from carrot.data.robotwin import robotwin_preprocess
from carrot.models.pi05.inference import Pi05Policy, create_trained_policy, policy_config
from carrot.models.pi05.inference.aloha_policy import AlohaOutputs
from carrot.models.pi05.inference.transforms import AbsoluteActions, Unnormalize
from carrot.models.pi05.loss_fn import Pi05SFTLossFn


class _Tokenizer:
    def __init__(self) -> None:
        self.prompts = []

    def __call__(self, prompts: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        self.prompts = prompts
        return {
            "input_ids": torch.arange(200)[None].expand(len(prompts), -1),
            "attention_mask": torch.ones(len(prompts), 200),
        }


class _Model(nn.Module):
    def __init__(self, actions: torch.Tensor) -> None:
        super().__init__()
        self.config = SimpleNamespace(action_dim=32, action_horizon=3)
        self.action_in_proj = nn.Linear(32, 1)
        self.actions = actions
        self.seen = None
        self.noise = None

    def sample_actions(
        self,
        device: torch.device,
        observation: Any,
        noise: torch.Tensor | None = None,
        num_steps: int = 10,
    ) -> torch.Tensor:
        self.seen = observation
        self.noise = noise
        assert num_steps == 10
        assert not self.training
        assert not torch.is_grad_enabled()
        return self.actions

    def forward(
        self, observation: Any, actions: torch.Tensor, noise: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        self.seen = observation
        return (actions - noise).square()


def _stats() -> dict[str, list[float]]:
    return {"q01": [-2.0] * 14, "q99": [3.0] * 14}


def _obs() -> dict[str, Any]:
    return {
        "state": np.linspace(-0.2, 0.8, 14, dtype=np.float32),
        "images": {
            "cam_high": np.full((3, 32, 64), 1, dtype=np.uint8),
            "cam_left_wrist": np.full((3, 24, 24), 127, dtype=np.uint8),
            "cam_right_wrist": np.full((3, 24, 24), 255, dtype=np.uint8),
        },
        "prompt": "  pick_up\nthe cup ",
    }


def test_infer_matches_training_observation_and_decodes_known_action() -> None:
    # 同一观测走训练与推理必须逐元素一致；常量 oracle 检查 delta、符号和夹爪的逆变换。
    obs = _obs()
    original = deepcopy(obs)
    tokenizer = _Tokenizer()
    model = _Model(torch.zeros(1, 3, 32))
    stats = _stats()
    policy = Pi05Policy(model, tokenizer, {"state": stats, "actions": stats}, device="cpu")
    noise = np.ones((3, 32), dtype=np.float32)

    # 归一化值零对应区间中点 0.5000005；手工计算实际执行动作，不用正向代码生成 oracle。
    result = policy.infer(obs, noise=noise)
    seen = model.seen
    prompts = tokenizer.prompts.copy()
    expected = np.broadcast_to(obs["state"], (3, 14)).copy()
    flip = np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])
    expected += 0.5000005 * flip
    expected[:, [6, 13]] = (0.5000005 + 0.5476 + 0.6213) / (1.4910 + 0.6213)
    np.testing.assert_allclose(result["actions"], expected, rtol=1e-6, atol=1e-6)
    assert result["actions"].shape == (3, 14)
    assert result["actions"].dtype == np.float32
    assert result["policy_timing"]["infer_ms"] > 0
    torch.testing.assert_close(model.noise, torch.ones(1, 3, 32), rtol=0, atol=0)

    # LeRobot 图像本来就是 [0,1] float；用极暗 uint8 图像暴露推理端的量纲歧义。
    image_keys = tuple(obs["images"])
    batch = {key: torch.tensor(value)[None].float() / 255 for key, value in obs["images"].items()}
    batch.update(
        {
            "observation.state": torch.tensor(obs["state"])[None],
            "action": torch.zeros(1, 3, 14),
            "task": [obs["prompt"]],
        }
    )
    loss = Pi05SFTLossFn(
        tokenizer,
        state_stats=stats,
        action_stats=stats,
        image_keys=image_keys,
        preprocess=robotwin_preprocess,
    )
    loss(model, batch)

    # 比较 adapter 输出；训练模型内部的随机增强不属于共享确定性预处理。
    for key in seen.images:
        torch.testing.assert_close(seen.images[key], model.seen.images[key], rtol=0, atol=0)
        assert seen.images[key].shape == (1, 3, 224, 224)
        torch.testing.assert_close(seen.image_masks[key], model.seen.image_masks[key])
    torch.testing.assert_close(seen.state, model.seen.state, rtol=0, atol=0)
    torch.testing.assert_close(seen.tokenized_prompt, model.seen.tokenized_prompt)
    assert prompts == tokenizer.prompts
    assert seen.state.shape == (1, 32)
    assert len(prompts[0].split("State: ")[1].split(";")[0].split()) == 32
    np.testing.assert_array_equal(obs["state"], original["state"])
    for key in obs["images"]:
        np.testing.assert_array_equal(obs["images"][key], original["images"][key])
    np.testing.assert_array_equal(noise, np.ones((3, 32)))


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_robotwin_action_round_trip(backend: str) -> None:
    # 多 batch、多 timestep 验证关节 delta 基准；夹爪动作与 state 使用不同转换，不能混用。
    state = np.arange(28, dtype=np.float32).reshape(2, 14) / 30
    actions = np.arange(84, dtype=np.float32).reshape(2, 3, 14) / 90
    if backend == "torch":
        state, actions = torch.tensor(state), torch.tensor(actions)
    original_state = np.asarray(state).copy()
    original_actions = np.asarray(actions).copy()

    # 正向转换后，将关节 delta 加回转换后的 state，再恢复执行命令。
    converted_state, converted_actions = robotwin_preprocess(state, actions)
    decoded = AlohaOutputs()(
        AbsoluteActions()(
            {
                "state": torch.as_tensor(converted_state),
                "actions": torch.as_tensor(converted_actions),
            }
        )
    )["actions"]

    # float32 连续变换只允许舍入误差，且不得修改调用方数据。
    np.testing.assert_allclose(decoded.numpy(), original_actions, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(state), original_state)
    np.testing.assert_array_equal(np.asarray(actions), original_actions)


def test_quantile_round_trip_and_padding() -> None:
    # 零跨度统计量依赖 epsilon；补齐的 18 维不能参与机器人反归一化。
    stats = {"q01": [2.0] * 14, "q99": [2.0] * 14}
    value = torch.full((1, 3, 14), 2.0)
    normalized = Pi05SFTLossFn._normalize(value, stats)
    padded = Pi05SFTLossFn._pad_last(normalized, 32)

    # 反归一化必须复用训练的 epsilon，保留尾部 padding。
    restored = Unnormalize({"actions": stats})({"actions": padded})["actions"]
    torch.testing.assert_close(restored[..., :14], value, rtol=0, atol=0)
    assert torch.count_nonzero(restored[..., 14:]) == 0


def test_tokenization_preserves_training_bin_boundaries() -> None:
    # 现有训练在边界处选择左区间；避免改成 np.digitize 后使已有 checkpoint 输入漂移。
    tokenizer = _Tokenizer()
    loss = Pi05SFTLossFn(
        tokenizer,
        state_stats=_stats(),
        action_stats=_stats(),
        image_keys=("high", "left", "right"),
    )

    # -1、0、1 包含精确边界，token 仍补到 200 位。
    tokens, masks = loss._tokenize("pick_up\ncup", torch.tensor([[-1.0, 0.0, 1.0]]))
    assert tokenizer.prompts == ["Task: pick up cup, State: -1 127 255;\nAction: "]
    assert tokens.shape == masks.shape == (1, 200)
    assert tokens.dtype == torch.long
    assert masks.dtype == torch.bool


@pytest.mark.parametrize(
    "missing",
    [
        "model.safetensors",
        "config.json",
        "norm_stats.json",
        "tokenizer_config.json",
    ],
)
def test_loader_rejects_missing_artifact_before_model_load(tmp_path: Path, missing: str) -> None:
    # 缺失文件必须先失败，不能触发大模型分配或联网 fallback。
    for name in ("model.safetensors", "config.json", "norm_stats.json", "tokenizer_config.json"):
        if name != missing:
            (tmp_path / name).touch()

    # 检查报错包含具体缺失文件，便于定位 checkpoint bundle 不完整。
    with pytest.raises(FileNotFoundError, match=missing):
        create_trained_policy(tmp_path, device="cpu")


@pytest.mark.parametrize("field", ["state", "image", "prompt", "noise"])
def test_infer_rejects_invalid_input(field: str) -> None:
    # 显式拒绝错误 schema，防止形状或图像量纲被自动猜测。
    obs = _obs()
    noise = None
    if field == "state":
        obs["state"][0] = np.nan
    elif field == "image":
        obs["images"]["cam_high"] = np.zeros((32, 64, 3), dtype=np.uint8)
    elif field == "prompt":
        obs["prompt"] = ""
    else:
        noise = np.zeros((3, 14), dtype=np.float32)
    model = _Model(torch.zeros(1, 3, 32))
    policy = Pi05Policy(model, _Tokenizer(), {"state": _stats(), "actions": _stats()}, device="cpu")

    # 所有错误都应在模型采样之前被发现。
    with pytest.raises(ValueError):
        policy.infer(obs, noise=noise)
    assert model.seen is None


@pytest.mark.parametrize("checkpoint_tokenizer", [True, False])
def test_loader_uses_checkpoint_stats_and_local_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpoint_tokenizer: bool
) -> None:
    # loader 必须使用 checkpoint 的 action 统计量，并优先读取随 checkpoint 保存的 tokenizer。
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    fallback = tmp_path / "tokenizer"
    fallback.mkdir()
    (fallback / "tokenizer_config.json").write_text("{}")
    if checkpoint_tokenizer:
        (checkpoint / "tokenizer_config.json").write_text("{}")
    for name in ("model.safetensors", "config.json"):
        (checkpoint / name).touch()
    (checkpoint / "norm_stats.json").write_text(
        json.dumps(
            {
                "state": _stats(),
                "action": {"q01": [0.0] * 14, "q99": [4.0] * 14},
            }
        )
    )
    calls = {}
    model = _Model(torch.zeros(1, 3, 32))

    # 用轻量替身捕获加载路径与联网选项，不冒充真实权重 smoke。
    def load_tokenizer(path: Path, **kwargs: Any) -> _Tokenizer:
        calls["tokenizer"] = path
        assert kwargs == {"local_files_only": True, "fix_mistral_regex": True}
        return _Tokenizer()

    def load_model(path: Path) -> _Model:
        calls["model"] = path
        return model

    monkeypatch.setattr(policy_config.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(policy_config.PI0Policy, "from_pretrained", load_model)
    policy = create_trained_policy(
        checkpoint, device="cpu", tokenizer_path=fallback, default_prompt="pick cup"
    )
    obs = _obs()
    del obs["prompt"]
    result = policy.infer(obs)

    # 零输出解码到 action 区间中点，而不是 state 的区间中点。
    assert calls["model"] == checkpoint
    assert calls["tokenizer"] == (checkpoint if checkpoint_tokenizer else fallback)
    np.testing.assert_allclose(
        result["actions"][:, 0], obs["state"][0] + 2.0000005, rtol=1e-6, atol=1e-6
    )
    assert policy.metadata["action_dim"] == 14


@pytest.mark.parametrize("invalid", ["nan", "reversed", "width"])
def test_invalid_stats_fail_before_model_setup(invalid: str) -> None:
    # 错误 quantile 不能被广播或静默兜底，否则输出仍 finite 却使用错误动作尺度。
    stats = _stats()
    if invalid == "nan":
        stats["q01"][0] = float("nan")
    elif invalid == "reversed":
        stats["q01"][0] = 4.0
    else:
        stats["q01"] = stats["q01"][:-1]

    # constructor 在模型转移设备之前校验统计量。
    with pytest.raises(ValueError, match="quantiles"):
        Pi05Policy(
            _Model(torch.zeros(1, 3, 32)),
            _Tokenizer(),
            {"state": stats, "actions": _stats()},
            device="cpu",
        )

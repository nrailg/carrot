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
from carrot.models.pi05.embodiments import (
    AlohaOutputs,
    create_aloha_transform_spec,
    create_libero_transform_spec,
)
from carrot.models.pi05.inference import (
    Pi05Policy,
    create_libero_policy,
    create_robotwin_policy,
    policy_config,
)
from carrot.models.pi05.loss_fn import Pi05SFTLossFn
from carrot.models.pi05.transforms import (
    AbsoluteActions,
    Pi05TransformSpec,
    Unnormalize,
    compose,
)


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
    def __init__(self, actions: torch.Tensor, *, action_horizon: int = 3) -> None:
        super().__init__()
        self.config = SimpleNamespace(action_dim=32, action_horizon=action_horizon)
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


class _RecordTransform:
    def __init__(self, events: list[str], name: str) -> None:
        self.events = events
        self.name = name

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        self.events.append(self.name)
        return data


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


def _robotwin_policy(
    model: _Model,
    tokenizer: _Tokenizer,
    stats: dict[str, list[float]],
    *,
    default_prompt: str | None = None,
) -> Pi05Policy:
    spec = create_aloha_transform_spec(
        tokenizer,
        {"state": stats, "actions": stats},
        model_action_dim=model.config.action_dim,
        default_prompt=default_prompt,
    )
    return Pi05Policy(model, spec, device="cpu")


def test_infer_matches_training_observation_and_decodes_known_action() -> None:
    # 同一观测走训练与推理必须逐元素一致；常量 oracle 检查 delta、符号和夹爪的逆变换。
    obs = _obs()
    original = deepcopy(obs)
    tokenizer = _Tokenizer()
    model = _Model(torch.zeros(1, 3, 32))
    stats = _stats()
    policy = _robotwin_policy(model, tokenizer, stats)
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
    restored = Unnormalize({"actions": stats}, use_quantiles=True)({"actions": padded})[
        "actions"
    ]
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
        create_robotwin_policy(tmp_path, device="cpu")


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
    policy = _robotwin_policy(model, _Tokenizer(), _stats())

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
    monkeypatch.setattr(policy_config.PI0Pytorch, "from_pretrained", load_model)
    policy = create_robotwin_policy(
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
        create_aloha_transform_spec(
            _Tokenizer(),
            {"state": stats, "actions": _stats()},
            model_action_dim=32,
        )


def _libero_stats() -> dict[str, dict[str, list[float]]]:
    return {
        "state": {"q01": [1.0] * 8, "q99": [5.0] * 8},
        "actions": {
            "q01": np.arange(7, dtype=np.float32).tolist(),
            "q99": (2 * np.arange(7, dtype=np.float32) + 2).tolist(),
        },
    }


def _libero_obs(*, include_actions: bool = False) -> dict[str, Any]:
    obs = {
        "observation/state": np.full(8, 3.0, dtype=np.float32),
        "observation/image": np.full((32, 48, 3), 255, dtype=np.uint8),
        "observation/wrist_image": np.zeros((32, 48, 3), dtype=np.uint8),
        "prompt": "pick_up\nthe cup",
    }
    if include_actions:
        obs["actions"] = np.zeros((10, 7), dtype=np.float32)
    return obs


def test_libero_transform_spec_is_reusable_for_training_samples() -> None:
    # 带 target actions 的样本必须通过与 inference 相同的 transform，锁定 Gate 4 复用边界。
    tokenizer = _Tokenizer()
    spec = create_libero_transform_spec(
        tokenizer,
        _libero_stats(),
        model_action_dim=32,
    )

    # 直接执行公开 input transforms，模拟训练 dataset 在 collate 前处理单条样本。
    transformed = compose(spec.inputs)(_libero_obs(include_actions=True))

    # 两路相机补为三路，缺失腕部相机必须保持 false mask，state/action 补到模型宽度。
    assert tuple(transformed["image"]) == (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )
    assert all(tuple(image.shape) == (3, 224, 224) for image in transformed["image"].values())
    assert transformed["image_mask"]["right_wrist_0_rgb"] is False
    assert torch.all(transformed["image"]["right_wrist_0_rgb"] == -1)
    assert transformed["state"].shape == (32,)
    assert transformed["actions"].shape == (10, 32)
    np.testing.assert_allclose(transformed["state"][:8], np.zeros(8), rtol=0, atol=1e-6)

    # LIBERO 的 PI0.5 prompt 不注入离散 state，避免错误复用 RoboTwin prompt contract。
    assert tokenizer.prompts == ["pick up the cup\n"]
    assert "State:" not in tokenizer.prompts[0]


def test_libero_policy_preserves_mask_and_decodes_seven_actions() -> None:
    # LIBERO 必须按官方 quantile contract 解码；失败意味着动作尺度仍与 checkpoint 不匹配。
    tokenizer = _Tokenizer()
    normalized_actions = torch.zeros(1, 10, 32)
    normalized_actions[..., :7] = torch.linspace(-1, 1, 7)
    model = _Model(normalized_actions, action_horizon=10)
    spec = create_libero_transform_spec(
        tokenizer,
        _libero_stats(),
        model_action_dim=32,
    )
    policy = Pi05Policy(model, spec, device="cpu")

    # 覆盖整个归一化区间，避免只检查中点而遗漏 quantile 缩放错误。
    result = policy.infer(_libero_obs(), noise=np.zeros((10, 32), dtype=np.float32))

    # 右腕图像存在但被 mask，最终动作必须 finite 且严格为 horizon=10、action_dim=7。
    assert not model.seen.image_masks["right_wrist_0_rgb"].item()
    assert torch.all(model.seen.images["right_wrist_0_rgb"] == -1)
    assert model.seen.state.shape == (1, 32)
    assert model.seen.tokenized_prompt.shape == (1, 200)
    assert result["actions"].shape == (10, 7)
    assert result["actions"].dtype == np.float32
    stats = _libero_stats()["actions"]
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    normalized = np.linspace(-1, 1, 7, dtype=np.float32)
    expected = (normalized + 1) / 2 * (q99 - q01 + 1e-6) + q01
    np.testing.assert_allclose(
        result["actions"], np.broadcast_to(expected, (10, 7)), rtol=1e-6, atol=1e-6
    )


def test_policy_executes_injected_transforms_in_declared_order() -> None:
    # 通用 executor 只能顺序执行 spec，不能按 embodiment 名称插入隐式分支。
    events = []
    tokenizer = _Tokenizer()
    model = _Model(torch.zeros(1, 10, 32), action_horizon=10)
    base = create_libero_transform_spec(tokenizer, _libero_stats(), model_action_dim=32)
    spec = Pi05TransformSpec(
        inputs=(
            _RecordTransform(events, "input:first"),
            *base.inputs,
            _RecordTransform(events, "input:last"),
        ),
        outputs=(
            _RecordTransform(events, "output:first"),
            *base.outputs,
            _RecordTransform(events, "output:last"),
        ),
        action_dim=base.action_dim,
    )

    # 一次 infer 应完整穿过两条 transform 链，且每个 transform 只执行一次。
    Pi05Policy(model, spec, device="cpu").infer(_libero_obs())

    # 事件顺序直接锁定 executor contract，未来新增 embodiment 不得改变它。
    assert events == ["input:first", "input:last", "output:first", "output:last"]


def test_libero_loader_reads_openpi_stats_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # loader 必须读取官方 assets 路径和 norm_stats wrapper，不能要求 Carrot 私有 JSON 布局。
    checkpoint = tmp_path / "checkpoint"
    stats_dir = checkpoint / "assets" / "physical-intelligence" / "libero"
    stats_dir.mkdir(parents=True)
    for name in ("model.safetensors", "config.json", "tokenizer_config.json"):
        (checkpoint / name).touch()
    (stats_dir / "norm_stats.json").write_text(json.dumps({"norm_stats": _libero_stats()}))
    model = _Model(torch.zeros(1, 10, 32), action_horizon=10)
    calls = {}

    # 用轻量替身确认 loader 路径；真实 checkpoint sampling 留给独立 GPU smoke。
    def load_tokenizer(path: Path, **kwargs: Any) -> _Tokenizer:
        calls["tokenizer"] = path
        assert kwargs == {"local_files_only": True, "fix_mistral_regex": True}
        return _Tokenizer()

    def load_model(path: Path) -> _Model:
        calls["model"] = path
        return model

    monkeypatch.setattr(policy_config.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(policy_config.PI0Pytorch, "from_pretrained", load_model)
    policy = create_libero_policy(checkpoint, device="cpu")

    # metadata 来自 LIBERO spec，证明 loader 没有落回 RoboTwin 14D contract。
    assert calls == {"tokenizer": checkpoint, "model": checkpoint}
    assert policy.metadata == {"action_horizon": 10, "action_dim": 7, "num_steps": 10}

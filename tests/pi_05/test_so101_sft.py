"""SO101 dataset and PI0.5 training/inference contract."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import default_collate

from carrot.data import so101
from carrot.models.pi05 import loss_fn as loss_fn_module
from carrot.models.pi05.embodiments.so101 import create_so101_transform_spec
from carrot.models.pi05.inference import policy_config
from carrot.models.pi05.inference.policy import Pi05Policy
from carrot.models.pi05.loss_fn import Pi05SFTLossFn, Pi05TransformedDataset


class _Tokenizer:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompts: list[str], **_: Any) -> dict[str, torch.Tensor]:
        self.prompts = prompts
        return {
            "input_ids": torch.ones(len(prompts), 200, dtype=torch.long),
            "attention_mask": torch.ones(len(prompts), 200, dtype=torch.long),
        }


class _Source:
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index != 0:
            raise IndexError(index)
        return {
            "observation.state": torch.full((6,), 2.0),
            "observation.images.top": torch.full((3, 64, 96), 255, dtype=torch.uint8),
            "observation.images.fpv": torch.zeros((3, 48, 80), dtype=torch.uint8),
            "task": "pick orange cube",
            "action": torch.full((3, 6), 5.0),
            "action_is_pad": torch.tensor([False, False, True]),
        }


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(action_dim=32, action_horizon=3, discrete_state_input=True)
        self.action_in_proj = nn.Linear(32, 4)

    def sample_actions(
        self,
        device: torch.device,
        observation: Any,
        noise: torch.Tensor | None = None,
        num_steps: int = 10,
    ) -> torch.Tensor:
        return torch.zeros((observation.state.shape[0], 3, 32), device=device)

    def forward(
        self,
        observation: Any,
        actions: torch.Tensor,
        noise: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        return (actions - noise).square()


def _stats() -> dict[str, dict[str, list[float]]]:
    return {
        "state": {"q01": [0.0] * 6, "q99": [10.0] * 6},
        "actions": {"q01": [0.0] * 6, "q99": [10.0] * 6},
    }


def test_so101_loader_maps_orange_cube_metadata_and_sample(monkeypatch) -> None:
    # 用公开数据集的字段形状构造元数据，防止工厂丢失相机、帧率或 padding 契约。
    metadata = SimpleNamespace(
        robot_type="so101_follower",
        fps=30,
        features={
            "observation.state": {"shape": [6]},
            "action": {"shape": [6]},
            "observation.images.top": {"dtype": "video", "shape": [540, 960, 3]},
            "observation.images.fpv": {"dtype": "video", "shape": [640, 480, 3]},
        },
        stats={
            "observation.state": _stats()["state"],
            "action": _stats()["actions"],
        },
    )
    captured: dict[str, Any] = {}

    def fake_metadata(repo_id: str, **kwargs: Any) -> Any:
        captured["metadata"] = (repo_id, kwargs)
        return metadata

    def fake_dataset(repo_id: str, **kwargs: Any) -> _Source:
        captured["dataset"] = (repo_id, kwargs)
        return _Source()

    # 将 LeRobot I/O 替换为单条样本，验证 revision、未来帧窗口和原始字段映射。
    monkeypatch.setattr(so101, "LeRobotDatasetMetadata", fake_metadata)
    monkeypatch.setattr(so101, "LeRobotDataset", fake_dataset)
    spec = so101.build_dataset(revision="pinned", action_horizon=3)
    sample = spec.dataset[0]

    # 动作保持绝对 6D，语言和 pad mask 直接来自 LeRobot 样本。
    assert captured["metadata"] == (
        "felixmayor/orange_cube_merged",
        {"root": None, "revision": "pinned"},
    )
    assert captured["dataset"][1]["delta_timestamps"] == {
        "action": [0.0, 1 / 30, 2 / 30]
    }
    assert captured["dataset"][1]["return_uint8"] is True
    assert spec.embodiment == "so101"
    assert sample["observation/image"].shape == (3, 64, 96)
    assert sample["observation/wrist_image"].shape == (3, 48, 80)
    assert sample["prompt"] == "pick orange cube"
    torch.testing.assert_close(sample["actions"], torch.full((3, 6), 5.0))
    torch.testing.assert_close(sample["action_is_pad"], torch.tensor([False, False, True]))


def test_so101_training_and_policy_share_absolute_action_contract() -> None:
    # 两路不同分辨率图像和 6D 绝对动作应能经过同一 SO101 输入变换。
    tokenizer = _Tokenizer()
    stats = _stats()
    spec = create_so101_transform_spec(tokenizer, stats, model_action_dim=32)
    source = so101.SO101SFTDataset(_Source())
    raw = source[0]
    batch = default_collate([Pi05TransformedDataset(source, spec)[0]])
    model = _Model()
    loss_fn = Pi05SFTLossFn(
        tokenizer,
        state_stats=stats["state"],
        action_stats=stats["actions"],
        image_keys=("observation.images.top", "observation.images.fpv"),
        transform_spec=spec,
    )

    # 比较训练与推理的图像、mask、状态和 token，确保第三视角确实无效。
    policy = Pi05Policy(model, spec, device="cpu")
    inference = policy._to_observation(policy._input_transform(raw))
    training, actions, valid = loss_fn.prepare_inputs(model, batch)
    for name in inference.images:
        torch.testing.assert_close(training.images[name], inference.images[name], rtol=0, atol=0)
        torch.testing.assert_close(training.image_masks[name], inference.image_masks[name])
    torch.testing.assert_close(training.state, inference.state, rtol=0, atol=0)
    torch.testing.assert_close(training.tokenized_prompt, inference.tokenized_prompt)
    assert not training.image_masks["right_wrist_0_rgb"].any()
    assert all(image.shape[-2:] == (224, 224) for image in training.images.values())
    assert tokenizer.prompts[0].startswith("Task: pick orange cube, State:")

    # 绝对动作 5 经 quantile 归一化为零，尾部补零且末帧不参与 loss。
    assert actions.shape == (1, 3, 32)
    torch.testing.assert_close(actions, torch.zeros_like(actions), rtol=0, atol=1e-6)
    torch.testing.assert_close(valid[0], torch.tensor([True, True, False]))
    loss, metrics = loss_fn(model, batch)
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, metrics["per_step_loss"][0, :2].mean())

    # 零归一化输出必须还原为 5，而不是加到当前 state 上的 delta。
    result = policy.infer({key: value for key, value in raw.items() if key != "actions"})
    np.testing.assert_allclose(result["actions"], 5.0, rtol=0, atol=1e-5)
    assert result["actions"].shape == (3, 6)


def test_so101_build_uses_configured_dataset_stats_instead_of_base_checkpoint(
    tmp_path, monkeypatch
) -> None:
    # 配置选用数据集统计量时，不能误读基座 checkpoint 中的 RoboTwin 14D 统计量。
    (tmp_path / "norm_stats.json").write_text(
        json.dumps(
            {
                "state": {"q01": [0.0] * 14, "q99": [1.0] * 14},
                "action": {"q01": [0.0] * 14, "q99": [1.0] * 14},
            }
        )
    )
    dataset_spec = so101.SFTDatasetSpec(
        dataset=so101.SO101SFTDataset(_Source()),
        collate_fn=None,
        state_stats=_stats()["state"],
        action_stats=_stats()["actions"],
        image_keys=("observation.images.top", "observation.images.fpv"),
        state_key="observation/state",
        action_key="actions",
        task_key="prompt",
        embodiment="so101",
    )

    # 不加载权重和视频，只经过真实的 build_pi05 分派和逐样本变换入口。
    monkeypatch.setattr(loss_fn_module, "load_callable", lambda path: lambda **kwargs: dataset_spec)
    monkeypatch.setattr(loss_fn_module.PI0Pytorch, "from_pretrained", lambda path: _Model())
    monkeypatch.setattr(
        loss_fn_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _Tokenizer(),
    )
    components = loss_fn_module.build_pi05(
        model_path=str(tmp_path),
        tokenizer_path=str(tmp_path),
        dataset_factory="carrot.data.so101.build_dataset",
        dataset_factory_kwargs={},
        device="cpu",
        norm_stats_source="dataset",
        preprocess=None,
    )

    # 校验六维统计、SO101 transform 和默认 collate 组合均来自目标数据集。
    assert components.loss_fn.state_stats == dataset_spec.state_stats
    assert components.loss_fn.action_stats == dataset_spec.action_stats
    assert components.loss_fn.transform_spec.action_dim == 6
    assert components.collate_fn is None
    assert components.dataset[0]["actions"].shape == (3, 32)


def test_so101_export_loads_as_six_joint_policy(tmp_path, monkeypatch) -> None:
    # 按训练导出的 checkpoint 文件布局提供 SO101 统计量，锁定 policy 工厂的读取路径。
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text("{}")
    (tmp_path / "norm_stats.json").write_text(
        json.dumps({"state": _stats()["state"], "action": _stats()["actions"]})
    )
    monkeypatch.setattr(policy_config.PI0Pytorch, "from_pretrained", lambda path: _Model())
    monkeypatch.setattr(
        policy_config.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _Tokenizer(),
    )

    # 真实工厂装配输入/输出变换，输出应为六维绝对关节位置。
    policy = policy_config.create_so101_policy(tmp_path, device="cpu")
    raw = so101.SO101SFTDataset(_Source())[0]
    result = policy.infer({key: value for key, value in raw.items() if key != "actions"})
    assert policy.metadata["action_dim"] == 6
    assert result["actions"].shape == (3, 6)
    np.testing.assert_allclose(result["actions"], 5.0, rtol=0, atol=1e-5)

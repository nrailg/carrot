"""SO101 dataset and PI0.5 training/inference contract."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import default_collate

from carrot.data import so101
from carrot.data.so101_units import model_stats, to_model_units, to_robot_units
from carrot.models.pi05 import loss_fn as loss_fn_module
from carrot.models.pi05.embodiments.so101 import create_so101_transform_spec
from carrot.models.pi05.inference import policy_config
from carrot.models.pi05.inference.policy import Pi05Policy
from carrot.models.pi05.loss_fn import Pi05SFTLossFn, Pi05TransformedDataset

DATASET_CONFIGS = yaml.safe_load(Path(__file__).with_name("so101_sft.yaml").read_text())


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
    spec = so101.build_dataset(
        repo_id="local/orange_cube", revision="pinned", action_horizon=3,
        **DATASET_CONFIGS["orange_cube"],
    )
    sample = spec.dataset[0]

    # 动作保持绝对 6D，语言和 pad mask 直接来自 LeRobot 样本。
    assert captured["metadata"] == (
        "local/orange_cube",
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
    np.testing.assert_allclose(sample["actions"], [[5, 5, 5, 5, 5, 0.05]] * 3)
    torch.testing.assert_close(sample["action_is_pad"], torch.tensor([False, False, True]))


def test_so101_training_and_policy_share_absolute_action_contract() -> None:
    # 两路不同分辨率图像和 6D 绝对动作应能经过同一 SO101 输入变换。
    tokenizer = _Tokenizer()
    stats = {key: model_stats(value, use_degrees=False) for key, value in _stats().items()}
    spec = create_so101_transform_spec(tokenizer, stats, model_action_dim=32)
    source = so101.SO101SFTDataset(_Source(), **DATASET_CONFIGS["orange_cube"])
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
    # 固定1e-6归一化epsilon在0.1夹爪范围中带来约1e-5偏差。
    torch.testing.assert_close(actions, torch.zeros_like(actions), rtol=0, atol=2e-5)
    torch.testing.assert_close(valid[0], torch.tensor([True, True, False]))
    loss, metrics = loss_fn(model, batch)
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, metrics["per_step_loss"][0, :2].mean())

    # 零归一化输出必须还原为 5，而不是加到当前 state 上的 delta。
    result = policy.infer({key: value for key, value in raw.items() if key != "actions"})
    np.testing.assert_allclose(result["actions"], [[5, 5, 5, 5, 5, 0.05]] * 3, rtol=0, atol=1e-5)
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
        dataset=so101.SO101SFTDataset(_Source(), **DATASET_CONFIGS["orange_cube"]),
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
    policy = policy_config.create_so101_policy(tmp_path, device="cpu", joint_units="normalized")
    raw = so101.SO101SFTDataset(_Source(), **DATASET_CONFIGS["orange_cube"])[0]
    result = policy.infer({key: value for key, value in raw.items() if key != "actions"})
    assert policy.metadata["action_dim"] == 6
    assert result["actions"].shape == (3, 6)
    np.testing.assert_allclose(result["actions"], [[5, 5, 5, 5, 5, 0.05]] * 3, rtol=0, atol=1e-5)


def test_so101_single_wrist_camera_at_15_fps(monkeypatch) -> None:
    # 单腕相机15FPS录制应保留时间间隔，并在训练和推理中屏蔽不存在的视角。
    class WristSource(_Source):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            sample["observation.images.wrist"] = sample.pop("observation.images.fpv")
            sample.pop("observation.images.top")
            return sample

    metadata = SimpleNamespace(
        robot_type="so_follower", fps=15,
        features={
            "observation.state": {"shape": [6]}, "action": {"shape": [6]},
            "observation.images.wrist": {"dtype": "video", "shape": [48, 80, 3]},
        },
        stats={"observation.state": _stats()["state"], "action": _stats()["actions"]},
    )
    captured = {}

    def make_dataset(repo_id, **kwargs):
        captured.update(kwargs)
        return WristSource()

    # 显式禁用外部相机，不复制腕图像伪造第二视角。
    monkeypatch.setattr(so101, "LeRobotDatasetMetadata", lambda *args, **kwargs: metadata)
    monkeypatch.setattr(so101, "LeRobotDataset", make_dataset)
    dataset = so101.build_dataset(
        repo_id="local/knock_down_the_cylinder", action_horizon=3,
        **DATASET_CONFIGS["knock_down_the_cylinder"],
    )
    raw = dataset.dataset[0]
    assert captured["delta_timestamps"] == {"action": [0.0, 1 / 15, 2 / 15]}
    assert dataset.image_keys == ("observation.images.wrist",)
    assert "observation/image" not in raw

    # 同一变换在训练collate和policy入口得到一致的单相机mask，输出仍为六维绝对动作。
    transform = create_so101_transform_spec(
        _Tokenizer(), {"state": dataset.state_stats, "actions": dataset.action_stats},
        model_action_dim=32,
    )
    batch = default_collate([Pi05TransformedDataset(dataset.dataset, transform)[0]])
    policy = Pi05Policy(_Model(), transform, device="cpu")
    obs = policy._to_observation(policy._input_transform(raw))
    assert not obs.image_masks["base_0_rgb"].any()
    assert obs.image_masks["left_wrist_0_rgb"].all()
    assert not obs.image_masks["right_wrist_0_rgb"].any()
    for name in obs.image_masks:
        torch.testing.assert_close(batch["image_mask"][name], obs.image_masks[name])
    output = policy.infer({key: value for key, value in raw.items() if key != "actions"})
    np.testing.assert_allclose(output["actions"],
                               to_model_units(np.full((3, 6), 5, dtype=np.float32),
                                              use_degrees=True), rtol=0, atol=1e-5)


def test_degree_samples_and_stats_use_radians_and_gripper_fraction(monkeypatch) -> None:
    # 六轴使用不同尺度，防止整向量乘同一个角度系数或遗漏统计量/夹爪转换。
    joints = np.float32([-180, -90, 0, 90, 180, 50])
    raw_stats = {key: joints.copy() for key in ("mean", "min", "max", "q01", "q99")}
    raw_stats["std"] = np.float32([180, 90, 1, 90, 180, 50])
    raw_stats["count"] = [354]
    original = {key: np.array(value).copy() for key, value in raw_stats.items()}

    class DegreeSource(_Source):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            sample["observation.state"] = joints
            sample["action"] = np.tile(joints, (3, 1))
            return sample

    metadata = SimpleNamespace(
        robot_type="so_follower", fps=15,
        features={"observation.state": {"shape": [6]}, "action": {"shape": [6]},
                  "observation.images.top": {"dtype": "video", "shape": [64, 96, 3]},
                  "observation.images.fpv": {"dtype": "video", "shape": [48, 80, 3]}},
        stats={"observation.state": raw_stats, "action": raw_stats},
    )
    monkeypatch.setattr(so101, "LeRobotDatasetMetadata", lambda *args, **kwargs: metadata)
    monkeypatch.setattr(so101, "LeRobotDataset", lambda *args, **kwargs: DegreeSource())

    # 真实工厂同时处理样本与stats；只修改degree开关，不依赖隐式相机默认。
    config = {**DATASET_CONFIGS["orange_cube"], "recorded_in_degrees": True}
    dataset = so101.build_dataset(repo_id="local/degrees", action_horizon=3, **config)
    sample = dataset.dataset[0]
    expected = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi, 0.5]

    # float32角度换算采用1e-6绝对容差；源样本、源stats和count必须保持不变。
    np.testing.assert_allclose(sample["observation/state"], expected, rtol=0, atol=1e-6)
    np.testing.assert_allclose(sample["actions"], [expected] * 3, rtol=0, atol=1e-6)
    for stats in (dataset.state_stats, dataset.action_stats):
        for key in ("mean", "min", "max", "q01", "q99"):
            np.testing.assert_allclose(stats[key], expected, rtol=0, atol=1e-6)
        np.testing.assert_allclose(stats["std"], np.abs(to_model_units(original["std"],
                                                                      use_degrees=True)))
        assert stats["count"] == [354]
    for key in raw_stats:
        np.testing.assert_array_equal(raw_stats[key], original[key])
    np.testing.assert_allclose(to_robot_units(sample["actions"], use_degrees=True),
                               [joints] * 3, rtol=0, atol=2e-5)
    assert dataset.joint_units == "radians"
    assert dataset.dataset.recorded_in_degrees is True


def test_so101_factory_requires_explicit_repo_before_io(monkeypatch) -> None:
    # repo默认留空；缺少明确来源时不能触发LeRobot加载或Hub访问。
    def forbid_metadata(*args, **kwargs):
        pytest.fail("metadata I/O must not happen without an explicit repo_id")

    # 在相机、视频或统计量读取之前报告缺失repo。
    monkeypatch.setattr(so101, "LeRobotDatasetMetadata", forbid_metadata)
    with pytest.raises(AssertionError, match="repo_id explicitly"):
        so101.build_dataset(**DATASET_CONFIGS["orange_cube"])


def test_camera_keys_must_be_explicit() -> None:
    # 未配置任何视角必须立即失败，不能回退到top/fpv隐式字段。
    with pytest.raises(AssertionError, match="at least one camera"):
        so101.SO101SFTDataset(_Source())


def test_export_preserves_model_unit_metadata(tmp_path) -> None:
    # 导出的统计量必须记录radian和[0,1]夹爪，避免新客户端误连degree旧服务。
    tokenizer = _Tokenizer()
    tokenizer.save_pretrained = lambda path: None
    loss = Pi05SFTLossFn(tokenizer, state_stats=_stats()["state"],
                        action_stats=_stats()["actions"], image_keys=(), joint_units="radians")

    # 经过实际artifact导出入口，验证单位字段与六维统计量同时落盘。
    loss.save_artifacts(tmp_path)
    payload = json.loads((tmp_path / "norm_stats.json").read_text())
    assert payload["joint_units"] == "radians"
    assert payload["gripper_units"] == "fraction"
    assert payload["action"] == _stats()["actions"]


def test_legacy_degree_stats_and_new_export_use_same_model_units(tmp_path, monkeypatch) -> None:
    # 旧degree checkpoint只能显式迁移；新export必须直接读模型单位，不能重复转换。
    for name in ("model.safetensors", "tokenizer_config.json"):
        (tmp_path / name).touch()
    (tmp_path / "config.json").write_text("{}")
    stats_path = tmp_path / "norm_stats.json"
    legacy = {"state": _stats()["state"], "action": _stats()["actions"]}
    stats_path.write_text(json.dumps(legacy))
    monkeypatch.setattr(policy_config.PI0Pytorch, "from_pretrained", lambda path: _Model())
    monkeypatch.setattr(policy_config.AutoTokenizer, "from_pretrained",
                        lambda *args, **kwargs: _Tokenizer())
    config = {**DATASET_CONFIGS["orange_cube"], "recorded_in_degrees": True}
    sample = so101.SO101SFTDataset(_Source(), **config)[0]

    # 无声明时拒绝猜测；显式degree只在内存转换统计量，不修改历史checkpoint。
    with pytest.raises(AssertionError, match="require joint_units"):
        policy_config.create_so101_policy(tmp_path, device="cpu")
    legacy_policy = policy_config.create_so101_policy(tmp_path, device="cpu", joint_units="degrees")
    legacy_actions = legacy_policy.infer(sample)["actions"]
    expected = to_model_units(np.full((3, 6), 5, dtype=np.float32), use_degrees=True)
    np.testing.assert_allclose(legacy_actions, expected, rtol=0, atol=1e-5)
    assert legacy_policy.metadata["joint_units"] == "radians"
    assert legacy_policy.metadata["gripper_units"] == "fraction"
    assert json.loads(stats_path.read_text()) == legacy

    # 新统计量附带单位，按原值使用；冲突的旧单位override必须报错。
    converted = {key: model_stats(value, use_degrees=True) for key, value in legacy.items()}
    stats_path.write_text(json.dumps(converted | {"joint_units": "radians",
                                                 "gripper_units": "fraction"}))
    policy = policy_config.create_so101_policy(tmp_path, device="cpu")
    np.testing.assert_array_equal(policy.infer(sample)["actions"], legacy_actions)
    with pytest.raises(AssertionError, match="disagrees"):
        policy_config.create_so101_policy(tmp_path, device="cpu", joint_units="degrees")

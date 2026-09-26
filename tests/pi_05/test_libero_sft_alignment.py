"""LIBERO training samples must reuse the inference input contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, default_collate

from carrot.data import SFTDatasetSpec
from carrot.models.pi05 import loss_fn as loss_fn_module
from carrot.models.pi05.embodiments.libero import create_libero_transform_spec
from carrot.models.pi05.inference.policy import Pi05Policy
from carrot.models.pi05.loss_fn import Pi05SFTLossFn, Pi05TransformedDataset, build_pi05


class _Tokenizer:
    def __call__(self, prompts: list[str], **_: Any) -> dict[str, torch.Tensor]:
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
            "observation/state": np.full(8, 2, dtype=np.float32),
            "observation/image": np.full((3, 256, 256), 255, dtype=np.uint8),
            "observation/wrist_image": np.zeros((3, 256, 256), dtype=np.uint8),
            "prompt": "pick the cup",
            "actions": np.ones((10, 7), dtype=np.float32),
            "action_is_pad": torch.tensor([False] * 9 + [True]),
        }


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(action_dim=32, action_horizon=10)
        self.action_in_proj = nn.Linear(32, 4)

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
        "state": {"q01": [0.0] * 8, "q99": [4.0] * 8},
        "actions": {"q01": [0.0] * 7, "q99": [2.0] * 7},
    }


def test_libero_build_uses_configured_stats_file(tmp_path: Path, monkeypatch) -> None:
    # 配置路径必须决定 LIBERO 训练统计量，即使模型目录中另有根目录统计量。
    configured = _stats()
    stats_path = tmp_path / "configured.json"
    stats_path.write_text(json.dumps({"norm_stats": configured}))
    (tmp_path / "norm_stats.json").write_text(
        json.dumps({"state": {"q01": [0] * 14}, "action": {"q01": [0] * 14}})
    )
    dataset = SFTDatasetSpec(
        dataset=_Source(),
        collate_fn=None,
        state_stats={"q01": [0] * 8},
        action_stats={"q01": [0] * 7},
        image_keys=("observation.images.image", "observation.images.image2"),
        embodiment="libero",
    )

    # 隔离权重和 tokenizer 加载，只验证真实 build_pi05 的统计量分派与输入变换。
    monkeypatch.setattr(loss_fn_module, "load_callable", lambda _: lambda **kwargs: dataset)
    monkeypatch.setattr(loss_fn_module.PI0Pytorch, "from_pretrained", lambda _: _Model())
    monkeypatch.setattr(
        loss_fn_module.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: _Tokenizer()
    )
    components = build_pi05(
        model_path=str(tmp_path),
        tokenizer_path=str(tmp_path),
        dataset_factory="carrot.data.libero.build_dataset",
        dataset_factory_kwargs={},
        device="cpu",
        norm_stats_path=str(stats_path),
        preprocess=None,
    )

    # 根目录与数据集里的不兼容统计量均不能覆盖配置文件中的 8D/7D 来源。
    assert components.loss_fn.state_stats == configured["state"]
    assert components.loss_fn.action_stats == configured["actions"]
    assert components.dataset[0]["actions"].shape == (10, 32)

    # 未配置文件且 checkpoint 没有统计量时必须失败，不能回退到数据集统计量。
    with pytest.raises(FileNotFoundError, match="norm_stats.json"):
        build_pi05(
            model_path=str(tmp_path / "missing_checkpoint"),
            tokenizer_path=str(tmp_path),
            dataset_factory="carrot.data.libero.build_dataset",
            dataset_factory_kwargs={},
            device="cpu",
            preprocess=None,
        )


def test_libero_transforms_run_per_sample_before_loss_batch() -> None:
    # 数据集逐条执行 inference transform，loss 只组装 batch；padding 不参与平均损失。
    tokenizer = _Tokenizer()
    stats = _stats()
    spec = create_libero_transform_spec(tokenizer, stats, model_action_dim=32)
    dataset = Pi05TransformedDataset(_Source(), spec)
    batch = default_collate([dataset[0], dataset[0]])
    model = _Model()
    loss_fn = Pi05SFTLossFn(
        tokenizer,
        state_stats=stats["state"],
        action_stats=stats["actions"],
        image_keys=("base", "wrist"),
        transform_spec=spec,
    )

    # 同一原始样本经 policy 入口与 train batch 入口，图像、mask、state 和 token 必须相同。
    policy = Pi05Policy(model, spec, device="cpu")
    inference = policy._to_observation(policy._input_transform(_Source()[0]))
    training, actions, valid = loss_fn.prepare_inputs(model, batch)
    for name in inference.images:
        torch.testing.assert_close(
            training.images[name][0], inference.images[name][0], rtol=0, atol=0
        )
        torch.testing.assert_close(training.image_masks[name][0], inference.image_masks[name][0])
    torch.testing.assert_close(training.state[0], inference.state[0], rtol=0, atol=0)
    torch.testing.assert_close(training.tokenized_prompt[0], inference.tokenized_prompt[0])
    torch.testing.assert_close(
        training.tokenized_prompt_mask[0], inference.tokenized_prompt_mask[0]
    )

    # 7D target 按同一 quantile 归一化后补零到 32D，末帧 padding 被保留。
    assert actions.shape == (2, 10, 32)
    torch.testing.assert_close(
        actions[..., :7], torch.zeros_like(actions[..., :7]), rtol=0, atol=1e-6
    )
    torch.testing.assert_close(actions[..., 7:], torch.zeros_like(actions[..., 7:]), rtol=0, atol=0)
    assert valid.shape == (2, 10)
    assert not valid[:, -1].any()
    assert valid[:, :-1].all()

    # loss 保留既有噪声和时间抽样逻辑，单次前向结果必须有限。
    loss, metrics = loss_fn(model, batch)
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["per_step_loss"]).all()
    expected_loss = (metrics["per_step_loss"] * valid).sum() / valid.sum()
    torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)


def test_real_libero_sample_matches_inference_and_has_finite_loss() -> None:
    # 真实 LeRobot chunk 必须与 inference 逐模态同输入，并能进入官方权重的一次有限 loss。
    missing = [
        name
        for name in (
            "CARROT_PI05_LIBERO_DATASET_ROOT",
            "CARROT_PI05_LIBERO_CHECKPOINT",
            "CARROT_PI05_LIBERO_TOKENIZER",
        )
        if not os.environ.get(name)
    ]
    if missing:
        pytest.fail(f"missing required test assets: {', '.join(missing)}")
    if not torch.cuda.is_available():
        pytest.fail("real LIBERO sample test requires CUDA")
    root = os.environ["CARROT_PI05_LIBERO_DATASET_ROOT"]
    checkpoint = os.environ["CARROT_PI05_LIBERO_CHECKPOINT"]
    tokenizer_path = os.environ["CARROT_PI05_LIBERO_TOKENIZER"]
    components = build_pi05(
        model_path=checkpoint,
        tokenizer_path=tokenizer_path,
        dataset_factory="carrot.data.libero.build_dataset",
        dataset_factory_kwargs={"root": root, "action_horizon": 10},
        device="cuda",
        norm_stats_path=str(
            Path(checkpoint) / "assets/physical-intelligence/libero/norm_stats.json"
        ),
        preprocess=None,
    )
    dataset = components.dataset
    raw = dataset.source[0]
    # 使用训练器相同的 forkserver worker 边界，验证 tokenizer/transform 可跨进程取样。
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        multiprocessing_context="forkserver",
    )
    batch = next(iter(dataloader))
    model = components.model
    loss_fn = components.loss_fn
    policy = Pi05Policy(model, loss_fn.transform_spec, device="cuda")

    # policy 与 loss 入口使用同一 raw observation，逐项严格相等以锁定共享 transform。
    inference = policy._to_observation(policy._input_transform(raw))
    training, actions, valid = loss_fn.prepare_inputs(model, batch)
    for name in inference.images:
        torch.testing.assert_close(training.images[name], inference.images[name], rtol=0, atol=0)
        torch.testing.assert_close(
            training.image_masks[name], inference.image_masks[name], rtol=0, atol=0
        )
    torch.testing.assert_close(training.state, inference.state, rtol=0, atol=0)
    torch.testing.assert_close(
        training.tokenized_prompt, inference.tokenized_prompt, rtol=0, atol=0
    )
    torch.testing.assert_close(
        training.tokenized_prompt_mask, inference.tokenized_prompt_mask, rtol=0, atol=0
    )

    # 用官方 stats 独立计算 7D target；尾部 25 维必须为零，pad mask 与数据集一致。
    stats = loss_fn.action_stats
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    expected = 2 * (raw["actions"].numpy() - q01) / (q99 - q01 + 1e-6) - 1
    assert actions.shape == (1, 10, 32)
    np.testing.assert_allclose(actions[0, :, :7].cpu().numpy(), expected, rtol=0, atol=1e-6)
    assert torch.count_nonzero(actions[..., 7:]).item() == 0
    torch.testing.assert_close(valid[0].cpu(), ~raw["action_is_pad"])

    # 仅做一次真实模型前向，不做反向或 SFT 试跑；loss 和逐帧值必须有限。
    model.train()
    with torch.no_grad():
        loss, metrics = loss_fn(model, batch)
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["per_step_loss"]).all()
    print(f"real_libero_loss={loss.item():.9g}", flush=True)

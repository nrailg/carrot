import copy
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from carrot.trainer.sft.checkpoint import load_checkpoint, save_checkpoint


class _ExportableLinear(torch.nn.Linear):
    def save_pretrained(
        self, save_directory: Path, *, state_dict: dict[str, Any] | None = None
    ) -> None:
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        tensors = self.state_dict() if state_dict is None else state_dict
        save_file(
            {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()},
            str(directory / "model.safetensors"),
        )


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    # 验证模型使用 OpenPI 根目录格式保存，同时 optimizer DCP、scheduler 和 step 能完整恢复。
    # Arrange：先执行一次 AdamW 更新，确保 optimizer 已创建非空 moment 状态。
    checkpoint = tmp_path / "checkpoint"
    model = _ExportableLinear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    expected_optimizer = copy.deepcopy(optimizer.state_dict())

    # Act：保存统一目录，并检查模型不再嵌套在 pretrained_model 或模型 DCP 中。
    save_checkpoint(checkpoint, model, optimizer, scheduler, step=1)
    exported = load_file(checkpoint / "model.safetensors")
    assert (checkpoint / "optimizer" / ".metadata").is_file()
    assert not (checkpoint / "pretrained_model").exists()
    assert not (checkpoint / "dcp").exists()
    for name, value in expected.items():
        torch.testing.assert_close(exported[name], value)

    # Arrange：新建模型和空 optimizer，模拟进程重启后先读模型、再恢复训练状态。
    resumed_model = _ExportableLinear(2, 1)
    resumed_model.load_state_dict(exported)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, lambda _: 1.0)

    # Act：optimizer-only DCP 应按新模型参数重新绑定状态，并恢复 scheduler 与 step。
    step = load_checkpoint(
        checkpoint,
        resumed_optimizer,
        resumed_scheduler,
    )

    # Assert：模型来自 OpenPI 文件，训练状态来自独立 DCP，二者共同构成完整 resume。
    assert step == 1
    assert resumed_scheduler.state_dict() == scheduler.state_dict()
    for name, value in resumed_model.state_dict().items():
        torch.testing.assert_close(value, expected[name])
    actual_optimizer = resumed_optimizer.state_dict()
    assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
    for parameter_id, expected_state in expected_optimizer["state"].items():
        actual_state = actual_optimizer["state"][parameter_id]
        for name, expected_value in expected_state.items():
            torch.testing.assert_close(actual_state[name], expected_value)


def test_checkpoint_writes_model_artifacts(tmp_path: Path) -> None:
    # 验证 tokenizer、归一化统计等附属文件直接写入 OpenPI checkpoint 根目录。
    # Arrange：artifact writer 用最小 norm_stats 文件标记实际接收的导出路径。
    checkpoint = tmp_path / "checkpoint"
    model = _ExportableLinear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    def write_artifact(path: Path) -> None:
        (path / "norm_stats.json").write_text("{}")

    # Act：保存 checkpoint 并由 artifact writer 补齐非模型文件。
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        step=1,
        artifact_writer=write_artifact,
    )

    # Assert：附属文件必须与 model.safetensors 同级，便于目录直接作为 model.path。
    assert (checkpoint / "norm_stats.json").read_text() == "{}"

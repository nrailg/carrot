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
    model = _ExportableLinear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}

    save_checkpoint(tmp_path, model, optimizer, scheduler, step=1)
    exported = load_file(tmp_path / "pretrained_model" / "model.safetensors")
    for name, value in expected.items():
        torch.testing.assert_close(exported[name], value)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    step = load_checkpoint(tmp_path, model, optimizer, scheduler)

    assert step == 1
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name])

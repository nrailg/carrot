from pathlib import Path

import torch

from carrot.trainer.sft.checkpoint import load_checkpoint, save_checkpoint
from carrot.trainer.sft.trainer import _scheduler


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = _scheduler(optimizer, 0, 10)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}

    save_checkpoint(tmp_path, model, optimizer, scheduler, step=1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    step = load_checkpoint(tmp_path, model, optimizer, scheduler)

    assert step == 1
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name])

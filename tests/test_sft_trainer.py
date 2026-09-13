from pathlib import Path

import pytest
import torch
from torch import nn

from carrot.trainer.sft.config import SFTConfig
from carrot.trainer.sft.worker import SFTTrainWorker, _scheduler


def test_scheduler_reaches_configured_floor() -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = _scheduler(
        optimizer,
        warmup_steps=2,
        total_steps=6,
        decay_steps=6,
        decay_learning_rate=2.5e-6,
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4 / 3)
    for _ in range(6):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.5e-6)


def test_worker_teardown_does_not_wait_for_failed_peers(
    tmp_path: Path, monkeypatch
) -> None:
    destroyed = []
    monkeypatch.setattr("carrot.trainer.sft.worker.dist.is_initialized", lambda: True)
    monkeypatch.setattr(
        "carrot.trainer.sft.worker.dist.barrier",
        lambda: pytest.fail("teardown must not run a collective"),
    )
    monkeypatch.setattr(
        "carrot.trainer.sft.worker.dist.destroy_process_group",
        lambda: destroyed.append(True),
    )
    config = SFTConfig.from_dict(
        {
            "model": {"path": "model", "tokenizer_path": "tokenizer"},
            "output_dir": str(tmp_path),
        }
    )
    worker = SFTTrainWorker(config)

    worker.teardown()

    assert destroyed == [True]

from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from torch import nn

from carrot.trainer.sft.config import SFTConfig
from carrot.trainer.sft.trainer import SFTTrainer
from carrot.trainer.sft.worker import (
    SFTTrainWorker,
    _assert_fp32_optimizer_state,
    _scheduler,
)

_SMOKE_CONFIG = Path(__file__).resolve().parents[1] / "configs/smolvla_robotwin_sft_smoke.yaml"


def _config(tmp_path: Path, values: dict[str, Any] | None = None) -> SFTConfig:
    vlm = tmp_path / "vlm"
    vlm.mkdir(exist_ok=True)
    values = dict(values or {})
    model = dict(values.get("model") or {})
    model.setdefault("vlm_path", str(vlm))
    values["model"] = model
    return SFTConfig.from_dict(values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sft_trainer_runs_smolvla_on_ray(tmp_path: Path) -> None:
    values = yaml.safe_load(_SMOKE_CONFIG.read_text())
    if not Path(values["model"]["path"]).is_dir():
        pytest.skip("smolvla weights missing")
    values["output_dir"] = str(tmp_path)
    values["wandb"]["enabled"] = False
    values["save_freq"] = 0
    values["steps"] = 2
    config = SFTConfig.from_dict(values)

    results = SFTTrainer(config).run()

    assert len(results) == config.num_gpus
    assert all(result["step"] == 2 for result in results)
    assert all(torch.isfinite(torch.tensor(result["loss"])) for result in results)


def test_scheduler_matches_smolvla_warmup_and_floor() -> None:
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


def test_worker_teardown_does_not_wait_for_failed_peers(tmp_path: Path, monkeypatch) -> None:
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
    worker = SFTTrainWorker(_config(tmp_path))

    worker.teardown()

    assert destroyed == [True]


def test_optimizer_state_is_fp32_with_fp32_master() -> None:
    model = nn.Linear(1, 1).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.tensor([[1.0]])).sum()
    loss.backward()
    optimizer.step()

    _assert_fp32_optimizer_state(optimizer)
    state = optimizer.state[model.weight]
    assert state["exp_avg"].dtype is torch.float32
    assert state["exp_avg_sq"].dtype is torch.float32


def test_optimizer_state_rejects_bfloat16_master() -> None:
    model = nn.Linear(1, 1).bfloat16()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="master parameter"):
        _assert_fp32_optimizer_state(optimizer)

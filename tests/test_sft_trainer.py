from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from carrot.distributed import Worker
from carrot.trainer.sft.config import SFTConfig
from carrot.trainer.sft.trainer import SFTTrainer
from carrot.trainer.sft.worker import (
    SFTTrainWorker,
    SFTTrainWorkerImpl,
    _assert_fp32_optimizer_state,
    _scheduler,
)


def _config(tmp_path: Path, values: dict[str, Any] | None = None) -> SFTConfig:
    vlm = tmp_path / "vlm"
    vlm.mkdir(exist_ok=True)
    values = dict(values or {})
    model = dict(values.get("model") or {})
    model.setdefault("vlm_path", str(vlm))
    values["model"] = model
    return SFTConfig.from_dict(values)


class FakePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        loss = ((self.weight * batch - 0.0) ** 2).mean()
        return loss, {}


def _batch_to_cuda(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(device=torch.cuda.current_device())


class RaySFTWorker(Worker):
    def __init__(self, config: SFTConfig, resume: str | None = None) -> None:
        super().__init__()
        self.config = config
        self.impl: SFTTrainWorkerImpl | None = None

    def setup(self) -> None:
        torch.cuda.set_device(self.local_rank)
        model = FakePolicy().to(device=f"cuda:{self.local_rank}")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        self.impl = SFTTrainWorkerImpl(
            model=model,
            optimizer=optimizer,
            scheduler=_scheduler(optimizer, 0, self.config.steps),
            preprocessor=_batch_to_cuda,
            dataloader=DataLoader([torch.tensor([1.0])] * 4, batch_size=1),
            config=self.config,
        )

    def train(self) -> dict[str, float | int]:
        if self.impl is None:
            raise RuntimeError("SFT train worker is not set up")
        metrics = self.impl.train()
        metrics["rank"] = self.rank
        metrics["world_size"] = self.world_size
        return metrics


def test_sft_train_worker_impl_accumulates_gradients(tmp_path: Path) -> None:
    model = FakePolicy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = _config(
        tmp_path,
        {
            "steps": 2,
            "gradient_accumulation_steps": 2,
            "save_freq": 0,
            "log_freq": 10,
            "fsdp": {"enabled": False},
        },
    )
    worker_impl = SFTTrainWorkerImpl(
        model=model,
        optimizer=optimizer,
        scheduler=_scheduler(optimizer, 0, config.steps),
        preprocessor=lambda batch: batch,
        dataloader=DataLoader([torch.tensor([1.0])] * 4, batch_size=1),
        config=config,
    )

    metrics = worker_impl.train()

    assert metrics["step"] == 2
    assert model.weight.item() < 1.0


def test_fsdp_does_not_disable_gradient_sync_during_accumulation(tmp_path: Path) -> None:
    # 关掉 sync 会让 unsharded grad 驻留，OOM
    sync_calls: list[bool] = []

    class TrackingPolicy(FakePolicy):
        def set_requires_gradient_sync(self, sync: bool) -> None:
            sync_calls.append(sync)

        def clip_grad_norm_(self, max_norm: float):
            return torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)

    model = TrackingPolicy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = _config(
        tmp_path,
        {
            "steps": 1,
            "gradient_accumulation_steps": 2,
            "save_freq": 0,
            "log_freq": 10,
            "fsdp": {"enabled": True},
        },
    )
    worker_impl = SFTTrainWorkerImpl(
        model=model,
        optimizer=optimizer,
        scheduler=_scheduler(optimizer, 0, config.steps),
        preprocessor=lambda batch: batch,
        dataloader=DataLoader([torch.tensor([1.0])] * 2, batch_size=1),
        config=config,
    )

    metrics = worker_impl.train()

    assert metrics["step"] == 1
    assert sync_calls == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sft_trainer_runs_on_ray(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("carrot.trainer.sft.trainer.SFTTrainWorker", RaySFTWorker)
    config = _config(
        tmp_path,
        {
            "steps": 2,
            "gradient_accumulation_steps": 2,
            "save_freq": 0,
            "log_freq": 10,
            "num_gpus": 1,
            "dataset": {"num_workers": 0},
            "fsdp": {"enabled": False},
        },
    )

    results = SFTTrainer(config).run()

    assert len(results) == 1
    assert results[0]["step"] == 2
    assert results[0]["rank"] == 0
    assert results[0]["world_size"] == 1
    assert results[0]["loss"] < 1.0


def test_scheduler_matches_smolvla_warmup_and_floor() -> None:
    model = FakePolicy()
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
    model = FakePolicy().float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss, _ = model(torch.tensor([1.0]))
    loss.backward()
    optimizer.step()

    _assert_fp32_optimizer_state(optimizer)
    state = optimizer.state[model.weight]
    assert state["exp_avg"].dtype is torch.float32
    assert state["exp_avg_sq"].dtype is torch.float32


def test_optimizer_state_rejects_bfloat16_master() -> None:
    model = FakePolicy().bfloat16()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="master parameter"):
        _assert_fp32_optimizer_state(optimizer)

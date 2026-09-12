import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from carrot.trainer.sft.config import SFTConfig
from carrot.trainer.sft.trainer import SFTTrainer
from carrot.trainer.sft.worker import (
    SFTTrainWorker,
    SFTTrainWorkerImpl,
    _assert_fp32_optimizer_state,
    _scheduler,
)


class FakePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        loss = ((self.weight * batch - 0.0) ** 2).mean()
        return loss, {}


class FakeWorkerGroup:
    def __init__(self) -> None:
        self.method = None

    def call(self, method: str):
        self.method = method
        return self

    def wait(self):
        return [{"step": 1, "loss": 0.5}]


class FakeCluster:
    def __init__(self) -> None:
        self.reservation = None
        self.launch_args = None
        self.workers = FakeWorkerGroup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def reserve(self, name, spec):
        self.reservation = (name, spec)

    def launch(self, name, worker_cls, *args, **kwargs):
        self.launch_args = (name, worker_cls, args, kwargs)
        return self.workers


def test_sft_train_worker_impl_accumulates_gradients() -> None:
    model = FakePolicy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = SFTConfig.from_dict(
        {
            "steps": 2,
            "gradient_accumulation_steps": 2,
            "save_freq": 0,
            "log_freq": 10,
            "fsdp": {"enabled": False},
        }
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


def test_fsdp_does_not_disable_gradient_sync_during_accumulation() -> None:
    # 关掉 sync 会让 unsharded grad 驻留，OOM
    sync_calls: list[bool] = []

    class TrackingPolicy(FakePolicy):
        def set_requires_gradient_sync(self, sync: bool) -> None:
            sync_calls.append(sync)

        def clip_grad_norm_(self, max_norm: float):
            return torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)

    model = TrackingPolicy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = SFTConfig.from_dict(
        {
            "steps": 1,
            "gradient_accumulation_steps": 2,
            "save_freq": 0,
            "log_freq": 10,
            "fsdp": {"enabled": True},
        }
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


def test_sft_trainer_controls_gpu_workers(monkeypatch) -> None:
    cluster = FakeCluster()
    monkeypatch.setattr("carrot.trainer.sft.trainer.Cluster", lambda **kwargs: cluster)
    config = SFTConfig.from_dict({"num_gpus": 2, "dataset": {"num_workers": 0}})

    results = SFTTrainer(config).run()

    assert results == [{"step": 1, "loss": 0.5}]
    assert cluster.reservation[0] == "sft"
    assert cluster.launch_args[0] == "sft-train-worker"
    assert cluster.launch_args[1] is SFTTrainWorker
    assert cluster.workers.method == "train"


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


def test_worker_teardown_does_not_wait_for_failed_peers(monkeypatch) -> None:
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
    worker = SFTTrainWorker(SFTConfig())

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

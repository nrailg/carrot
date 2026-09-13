import os
from pathlib import Path

import pytest
import torch
from torch import nn

from carrot.trainer.sft.config import SFTConfig
from carrot.trainer.sft.trainer import SFTTrainer
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


def test_pi05_sft_runs_two_steps_on_ray_gpu(tmp_path: Path) -> None:
    model_path = os.environ.get(
        "CARROT_PI05_MODEL_PATH",
        "/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/Miical/pi05-base",
    )
    dataset_root = os.environ.get(
        "CARROT_ROBOTWIN_ROOT",
        "/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/lerobot/robotwin_unified",
    )
    config = SFTConfig.from_dict(
        {
            "model": {"path": model_path, "tokenizer_path": model_path},
            "dataset": {
                "factory_kwargs": {
                    "repo_id": "lerobot/robotwin_unified",
                    "root": dataset_root,
                },
                "num_workers": 0,
            },
            "optimizer": {
                "learning_rate": 2.5e-5,
                "warmup_steps": 0,
                "decay_steps": 2,
                "decay_learning_rate": 2.5e-6,
            },
            "fsdp": {
                "enabled": True,
                "param_dtype": "bfloat16",
                "reduce_dtype": "float32",
            },
            "output_dir": str(tmp_path / "pi05-sft"),
            "steps": 2,
            "micro_batch_size": 1,
            "global_batch_size": 1,
            "log_freq": 1,
            "save_freq": 0,
            "dp_size": 1,
            "num_nodes": 1,
        }
    )

    results = SFTTrainer(config).run()

    assert len(results) == 1
    assert results[0]["step"] == 2
    assert torch.isfinite(torch.tensor(results[0]["loss"]))

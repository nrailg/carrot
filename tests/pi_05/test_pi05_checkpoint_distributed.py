import gc
import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import DTensor

from carrot.models.pi05.model import PI0Pytorch
from carrot.models.pi05.parallelize import Pi05Parallelizer
from carrot.parallel import FSDPConfig, parallelize_model
from carrot.trainer.sft.checkpoint import load_checkpoint, save_checkpoint

PI05_CHECKPOINT = os.environ.get("CARROT_PI05_OPENPI_PYTORCH_CHECKPOINT")


def _initialize_adamw_state(optimizer: torch.optim.AdamW) -> None:
    for param_group in optimizer.param_groups:
        for parameter in param_group["params"]:
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)
    learning_rates = [param_group["lr"] for param_group in optimizer.param_groups]
    for param_group in optimizer.param_groups:
        param_group["lr"] = 0.0
    optimizer.step()
    for param_group, learning_rate in zip(
        optimizer.param_groups,
        learning_rates,
        strict=True,
    ):
        param_group["lr"] = learning_rate
    optimizer.zero_grad(set_to_none=True)


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    if isinstance(value, DTensor):
        value = value.to_local()
    return value.detach()


def _optimizer_probe(optimizer: torch.optim.AdamW) -> dict[str, Any]:
    states = list(optimizer.state.values())
    first_state = states[0]
    return {
        "count": len(states),
        "step": _local_tensor(first_state["step"]).cpu().clone(),
        "exp_avg": _local_tensor(first_state["exp_avg"]).flatten()[:8].cpu().clone(),
        "exp_avg_sq": _local_tensor(first_state["exp_avg_sq"]).flatten()[:8].cpu().clone(),
    }


def _run_pi05_checkpoint_round_trip(
    rank: int,
    world_size: int,
    init_file: str,
    source_path: str,
    checkpoint_path: str,
) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    checkpoint = Path(checkpoint_path)
    try:
        # 使用真实 PI0.5 参数规模和模块层级建立与 SFT worker 相同的 FSDP2 拓扑。
        model = PI0Pytorch.from_pretrained(source_path).to(rank)
        parallelize_model(model, Pi05Parallelizer(), FSDPConfig())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-10)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        _initialize_adamw_state(optimizer)

        # 写入非默认探针，区分真正的 DCP 恢复与 load 前的零状态初始化。
        first_state = next(iter(optimizer.state.values()))
        _local_tensor(first_state["step"]).fill_(7)
        first_exp_avg = _local_tensor(first_state["exp_avg"]).flatten()
        if first_exp_avg.numel():
            first_exp_avg[0] = rank + 1
        expected_optimizer = _optimizer_probe(optimizer)

        save_checkpoint(checkpoint, model, optimizer, scheduler, step=123)
        if rank == 0:
            assert (checkpoint / "model.safetensors").is_file()
            assert (checkpoint / "optimizer" / ".metadata").is_file()
            with (checkpoint / "config.json").open() as stream:
                saved_config = json.load(stream)
            assert saved_config["precision"] == "float32"
        dist.barrier()

        del model, optimizer, scheduler, first_state, first_exp_avg
        gc.collect()
        torch.cuda.empty_cache()

        # checkpoint 根目录必须能被 PI0Pytorch 直接 strict load，再恢复 optimizer shard。
        resumed_model = PI0Pytorch.from_pretrained(checkpoint).to(rank)
        assert all(parameter.dtype is torch.float32 for parameter in resumed_model.parameters())
        parallelize_model(resumed_model, Pi05Parallelizer(), FSDPConfig())
        resumed_optimizer = torch.optim.AdamW(
            resumed_model.parameters(),
            lr=1e-5,
            weight_decay=1e-10,
        )
        resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(
            resumed_optimizer,
            lambda _: 1.0,
        )
        step = load_checkpoint(checkpoint, resumed_optimizer, resumed_scheduler)
        assert step == 123

        actual_optimizer = _optimizer_probe(resumed_optimizer)
        assert actual_optimizer["count"] == expected_optimizer["count"]
        torch.testing.assert_close(actual_optimizer["step"], expected_optimizer["step"])
        torch.testing.assert_close(
            actual_optimizer["exp_avg"],
            expected_optimizer["exp_avg"],
        )
        torch.testing.assert_close(
            actual_optimizer["exp_avg_sq"],
            expected_optimizer["exp_avg_sq"],
        )

        # 同一真实 PI0.5 resume 对象必须可以再次执行完整 checkpoint 保存。
        save_checkpoint(
            checkpoint.with_name("pi05-checkpoint-next"),
            resumed_model,
            resumed_optimizer,
            resumed_scheduler,
            step=124,
        )
    finally:
        dist.destroy_process_group()


def test_real_pi05_fsdp_checkpoint_round_trip(tmp_path: Path) -> None:
    # 该测试显式 opt-in，避免日常单测加载 3B 参数；Gemini 上使用本地 PI0.5 checkpoint。
    if not PI05_CHECKPOINT:
        pytest.fail("set CARROT_PI05_OPENPI_PYTORCH_CHECKPOINT to a PI0.5 checkpoint")
    if torch.cuda.device_count() < 2:
        pytest.fail("PI0.5 checkpoint round trip requires two CUDA devices")
    init_file = tmp_path / "process-group-init"
    checkpoint = tmp_path / "pi05-checkpoint"
    mp.spawn(
        _run_pi05_checkpoint_round_trip,
        args=(2, str(init_file), PI05_CHECKPOINT, str(checkpoint)),
        nprocs=2,
        join=True,
    )

import copy
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file, save_file
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from carrot.trainer.sft.checkpoint import load_checkpoint, save_checkpoint


class _ExportableLinear(torch.nn.Linear):
    def save_pretrained(
        self, save_directory: Path, *, state_dict: dict[str, Any] | None = None
    ) -> None:
        # checkpoint 层必须把原始 DTensor 交给模型；模型负责 collective materialization。
        assert state_dict is not None
        assert any(isinstance(tensor, DTensor) for tensor in state_dict.values())
        rank_zero = dist.get_rank() == 0
        tensors = {}
        for name, value in state_dict.items():
            tensor = value.full_tensor() if isinstance(value, DTensor) else value
            if rank_zero:
                tensors[name] = tensor.detach().cpu().contiguous()
            del tensor
        if not rank_zero:
            dist.barrier()
            return
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_file(
            tensors,
            str(directory / "model.safetensors"),
        )
        dist.barrier()


def _local_optimizer_state(optimizer: torch.optim.Optimizer) -> dict[int, dict[str, Any]]:
    state = optimizer.state_dict()["state"]
    return {
        parameter_id: {
            name: value.to_local().detach().clone()
            if isinstance(value, DTensor)
            else value.detach().clone()
            if isinstance(value, torch.Tensor)
            else copy.deepcopy(value)
            for name, value in parameter_state.items()
        }
        for parameter_id, parameter_state in state.items()
    }


def _run_fsdp_checkpoint_round_trip(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_path: str,
) -> None:
    # 每个子进程绑定独立 GPU，并通过文件 rendezvous 构造最小两卡 FSDP2 进程组。
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    checkpoint = Path(checkpoint_path)
    try:
        # 构造带非空 AdamW moment 的 FSDP2 模型，覆盖真实训练保存前的状态。
        torch.manual_seed(7)
        model = _ExportableLinear(4, 2, device=f"cuda:{rank}")
        fully_shard(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        inputs = torch.arange(8, device=f"cuda:{rank}", dtype=torch.float32).reshape(2, 4)
        model(inputs).sum().backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        expected_optimizer = _local_optimizer_state(optimizer)

        # 所有 rank 共同保存；模型应汇聚到根目录，optimizer 应保持 DCP shard。
        save_checkpoint(checkpoint, model, optimizer, scheduler, step=1)
        assert (checkpoint / "model.safetensors").is_file()
        assert (checkpoint / "optimizer" / ".metadata").is_file()

        # 在销毁旧训练对象前保存期望输出；optimizer 预期值已按各 rank 本地 shard 固化。
        expected_output = model(inputs).detach()
        del model, optimizer

        # 模拟新进程先加载 OpenPI 模型文件，再按相同拓扑重新建立 FSDP2 和 optimizer。
        resumed_model = _ExportableLinear(4, 2)
        resumed_model.load_state_dict(load_file(checkpoint / "model.safetensors"))
        resumed_model = resumed_model.to(rank)
        fully_shard(resumed_model)
        resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
        resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(
            resumed_optimizer,
            lambda _: 1.0,
        )

        # optimizer-only DCP 必须恢复 shard，trainer_state 必须恢复 scheduler 和 step。
        step = load_checkpoint(
            checkpoint,
            resumed_optimizer,
            resumed_scheduler,
        )
        assert step == 1
        assert resumed_scheduler.state_dict() == scheduler.state_dict()

        # 两个 FSDP2 模型在相同输入上的输出必须严格一致，证明根目录模型权重完整。
        torch.testing.assert_close(resumed_model(inputs), expected_output, rtol=0, atol=0)

        # 每个 rank 的本地 AdamW step 和两个 moment 必须与保存前的 shard 严格一致。
        actual_optimizer = resumed_optimizer.state_dict()["state"]
        assert actual_optimizer.keys() == expected_optimizer.keys()
        for parameter_id, expected_state in expected_optimizer.items():
            actual_state = actual_optimizer[parameter_id]
            for state_name, expected_value in expected_state.items():
                actual_value = actual_state[state_name]
                if isinstance(actual_value, DTensor):
                    actual_value = actual_value.to_local()
                torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)

        # 恢复后的对象必须还能继续保存下一份 checkpoint，防止 state-dict API 留下脏状态。
        save_checkpoint(
            checkpoint.with_name("checkpoint-next"),
            resumed_model,
            resumed_optimizer,
            resumed_scheduler,
            step=2,
        )
    finally:
        # 无论断言是否失败都销毁进程组，避免污染后续 GPU 测试。
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_fsdp_checkpoint_round_trip(tmp_path: Path) -> None:
    # 验证真实两卡 FSDP2 下，OpenPI 模型根目录与 optimizer-only DCP 能共同完成 resume。
    # Arrange：使用独立 rendezvous 文件和 checkpoint 目录隔离并行测试的共享状态。
    init_file = tmp_path / "process-group-init"
    checkpoint = tmp_path / "checkpoint"

    # Act/Assert：子进程内覆盖 collective 保存、重新分片加载及逐状态严格比较。
    mp.spawn(
        _run_fsdp_checkpoint_round_trip,
        args=(2, str(init_file), str(checkpoint)),
        nprocs=2,
        join=True,
    )

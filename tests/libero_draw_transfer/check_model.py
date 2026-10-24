import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from carrot.trainer.sft.config import SFTConfig
from carrot.trainer.sft.worker import SFTTrainWorker

GROUPS = {
    "vision": "paligemma.model.vision_tower",
    "vlm": "paligemma.model.language_model",
    "expert": "gemma_expert",
}


def local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = SFTConfig.from_yaml(args.config)
    cfg = replace(
        cfg,
        dataset=replace(cfg.dataset, num_workers=0),
        save_freq=0,
        wandb=replace(cfg.wandb, enabled=False),
    )
    assert int(os.environ["WORLD_SIZE"]) == cfg.dp_size == 8
    worker = SFTTrainWorker(cfg)
    worker.setup()
    try:
        impl = worker.impl
        assert impl is not None
        impl.model.train()
        # 三个参数组均须实际可训练；检查局部完整梯度，不以requires_grad声明代替证据。
        groups = {
            group: [(name, p) for name, p in impl.model.named_parameters() if marker in name]
            for group, marker in GROUPS.items()
        }
        assert all(values and all(p.requires_grad for _, p in values) for values in groups.values())
        iterator = iter(impl.dataloader)
        rows = []
        for step in range(2):
            # 与正式运行保持micro/GBS/GAS，数据和loss均走相同路径。
            impl.optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for _ in range(cfg.gas):
                loss, _ = impl.loss_fn(impl.model, next(iterator))
                assert torch.isfinite(loss)
                (loss / cfg.gas).backward()
                total += float(loss.detach()) / cfg.gas
            stats = {}
            snapshots = {}
            for group, parameters in groups.items():
                energy = torch.zeros((), device="cuda")
                for name, parameter in parameters:
                    if parameter.grad is not None:
                        grad = local(parameter.grad).detach().float()
                        assert torch.isfinite(grad).all(), name
                        energy += grad.square().sum()
                dist.all_reduce(energy)
                assert energy > 0, f"no gradient in {group}"
                # 首个非空参数的整个local shard用于确认更新，避免单个元素恰好梯度为零。
                candidates = [
                    (name, p) for name, p in parameters if p.grad is not None and local(p).numel()
                ]
                assert candidates, group
                name, parameter = candidates[0]
                snapshots[group] = (name, parameter, local(parameter).detach().clone())
                stats[group] = {"gradient_l2": float(energy.sqrt())}
            norm = torch.nn.utils.clip_grad_norm_(
                impl.model.parameters(), 1, error_if_nonfinite=True
            )
            impl.optimizer.step()
            for group, (name, parameter, before) in snapshots.items():
                difference = (local(parameter).detach().float() - before.float()).abs().max()
                dist.all_reduce(difference, op=dist.ReduceOp.MAX)
                assert difference > 0, f"parameter did not update: {name}"
                stats[group].update(parameter=name, update_max=float(difference))
            rows.append(
                {
                    "step": step + 1,
                    "rank0_loss": total,
                    "grad_norm": float(norm.full_tensor() if isinstance(norm, DTensor) else norm),
                    "groups": stats,
                }
            )
            if worker.rank == 0:
                print(json.dumps(rows[-1]), flush=True)
        # 两步后才写PASS；所有rank必须共同到达，局部成功不构成全局验收。
        dist.barrier()
        if worker.rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / "result.json").write_text(
                json.dumps({"passed": True, "steps": rows}, indent=2) + "\n"
            )
    finally:
        worker.teardown()


if __name__ == "__main__":
    main()

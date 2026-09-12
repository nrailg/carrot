"""Per-GPU SFT worker and its testable loop implementation."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import wandb
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from carrot.distributed import Worker
from carrot.modeling import parallelize_model
from carrot.models.smolvla import SmolVLAParallelizer, build_smolvla
from carrot.trainer.sft.checkpoint import load_checkpoint, save_checkpoint
from carrot.trainer.sft.config import SFTConfig


def _scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    *,
    decay_steps: int = 30_000,
    decay_learning_rate: float = 2.5e-6,
):
    """Build the warmup/cosine schedule used by LeRobot's SmolVLA preset."""
    if total_steps < decay_steps:
        scale_factor = total_steps / decay_steps
        warmup_steps = int(warmup_steps * scale_factor)
        decay_steps = total_steps
    peak_learning_rate = optimizer.param_groups[0]["lr"]

    def scale(step: int) -> float:
        if step < warmup_steps:
            if step <= 0:
                return 1 / (warmup_steps + 1)
            fraction = 1 - step / warmup_steps
            return (1 / (warmup_steps + 1) - 1) * fraction + 1
        bounded_step = min(step, decay_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * bounded_step / decay_steps))
        floor = decay_learning_rate / peak_learning_rate
        return (1 - floor) * cosine_decay + floor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _init_wandb(config: SFTConfig) -> Any:
    return wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.wandb.name,
        settings=wandb.Settings(console="off"),
        config={
            "model": config.model.path,
            "dataset": config.dataset.repo_id,
            "steps": config.steps,
            "batch_size": config.batch_size,
            "num_gpus": config.num_gpus,
            "num_nodes": config.num_nodes,
            "learning_rate": config.optimizer.learning_rate,
        },
    )


def _assert_fp32_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    """Fail fast unless AdamW masters and all floating-point state use FP32."""
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.dtype.is_floating_point and parameter.dtype is not torch.float32:
                raise ValueError(f"optimizer master parameter must be FP32, got {parameter.dtype}")
    for state in optimizer.state.values():
        for name, value in state.items():
            is_non_fp32_float = (
                torch.is_tensor(value)
                and value.dtype.is_floating_point
                and value.dtype is not torch.float32
            )
            if is_non_fp32_float:
                raise ValueError(f"optimizer state {name!r} must be FP32, got {value.dtype}")


class SFTTrainWorkerImpl:
    """Implementation detail that keeps the worker training loop maintainable."""
    # TODO 有点多余，应该合并到 SFTTrainWorker 中

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        preprocessor: Any,
        dataloader: DataLoader,
        config: SFTConfig,
        sampler: DistributedSampler | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.preprocessor = preprocessor
        self.dataloader = dataloader
        self.config = config
        self.sampler = sampler
        self.step = 0
        self.epoch = 0
        self._wandb = None
        self._optimizer_state_checked = False

    def train(self) -> dict[str, float | int]:
        self.model.train()
        mean_loss = float("nan")
        self.optimizer.zero_grad(set_to_none=True)
        if self._is_rank_0() and self.config.wandb.enabled:
            self._wandb = _init_wandb(self.config)
        if self._is_rank_0():
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            print(
                f"starting SFT loop steps={self.config.steps} "
                f"batch_size={self.config.batch_size} world={world_size}",
                flush=True,
            )
        iterator = iter(self.dataloader)
        try:
            return self._train_loop(iterator, mean_loss)
        finally:
            if self._wandb is not None:
                self._wandb.finish()
                self._wandb = None

    def _train_loop(self, iterator, mean_loss: float) -> dict[str, float | int]:
        while self.step < self.config.steps:
            accumulated_loss = 0.0
            for micro_step in range(self.config.gradient_accumulation_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    self.epoch += 1
                    if self.sampler is not None:
                        self.sampler.set_epoch(self.epoch)
                    iterator = iter(self.dataloader)
                    batch = next(iterator)
                if self._is_rank_0() and self.step == 0 and micro_step == 0:
                    print("first batch fetched, running forward", flush=True)
                batch = self.preprocessor(batch)
                loss, _ = self.model(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss at step {self.step}: {loss.item()}"
                    )
                scaled = loss / self.config.gradient_accumulation_steps
                if self._is_rank_0() and self.step == 0 and micro_step == 0:
                    print(
                        f"first forward ok loss={loss.detach().float().item():.6f} "
                        f"dtype={loss.dtype}",
                        flush=True,
                    )
                scaled.backward()
                accumulated_loss += loss.detach().float().item()

            if self.config.optimizer.max_grad_norm:
                if self.config.fsdp.enabled:
                    self.model.clip_grad_norm_(self.config.optimizer.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.optimizer.max_grad_norm
                    )
            self.optimizer.step()
            if not self._optimizer_state_checked:
                _assert_fp32_optimizer_state(self.optimizer)
                self._optimizer_state_checked = True
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1

            mean_loss = accumulated_loss / self.config.gradient_accumulation_steps
            if dist.is_initialized():
                value = torch.tensor(mean_loss, device=torch.cuda.current_device())
                dist.all_reduce(value)
                mean_loss = value.item() / dist.get_world_size()
            learning_rate = self.optimizer.param_groups[0]["lr"]
            if self._is_rank_0() and self.step % self.config.log_freq == 0:
                print(
                    f"step={self.step} loss={mean_loss:.6f} lr={learning_rate:.3e}",
                    flush=True,
                )
                if self._wandb is not None:
                    self._wandb.log(
                        {"train/loss": mean_loss, "train/lr": learning_rate},
                        step=self.step,
                    )
            if self.config.save_freq and self.step % self.config.save_freq == 0:
                save_checkpoint(
                    Path(self.config.output_dir) / "checkpoints" / f"step-{self.step:08d}",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.step,
                )
        if self.config.save_freq and self.step % self.config.save_freq != 0:
            save_checkpoint(
                Path(self.config.output_dir) / "checkpoints" / f"step-{self.step:08d}",
                self.model,
                self.optimizer,
                self.scheduler,
                self.step,
            )
        return {"step": self.step, "loss": mean_loss}

    @staticmethod
    def _is_rank_0() -> bool:
        return not dist.is_initialized() or dist.get_rank() == 0


class SFTTrainWorker(Worker):
    """One GPU-bound worker containing the complete SFT execution path."""

    def __init__(self, config: SFTConfig, resume: str | None = None) -> None:
        super().__init__()
        self.config = config
        self.resume = resume
        self.impl: SFTTrainWorkerImpl | None = None

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SmolVLA FSDP training requires CUDA")
        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend="nccl", rank=self.rank, world_size=self.world_size)
        random.seed(self.config.seed + self.rank)
        torch.manual_seed(self.config.seed + self.rank)
        torch.cuda.manual_seed_all(self.config.seed + self.rank)

        # TODO: avoid hardcoded model
        components = build_smolvla(
            model_path=self.config.model.path,
            dataset_repo_id=self.config.dataset.repo_id,
            dataset_root=self.config.dataset.root,
            device=f"cuda:{self.local_rank}",
            video_backend=self.config.dataset.video_backend,
            rename_map=self.config.dataset.rename_map or None,
            vlm_path=self.config.model.vlm_path,
        )
        model = components.policy
        parallelize_model(model, SmolVLAParallelizer(), self.config.fsdp)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("SmolVLA policy has no trainable parameters")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.config.optimizer.learning_rate,
            weight_decay=self.config.optimizer.weight_decay,
            betas=self.config.optimizer.betas,
            eps=self.config.optimizer.eps,
        )
        _assert_fp32_optimizer_state(optimizer)
        scheduler = _scheduler(
            optimizer,
            self.config.optimizer.warmup_steps,
            self.config.steps,
            decay_steps=self.config.optimizer.decay_steps,
            decay_learning_rate=self.config.optimizer.decay_learning_rate,
        )
        # TODO: avoid hardcoded dataset
        sampler = DistributedSampler(
            components.dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            seed=self.config.seed,
            drop_last=True,
        )
        dataloader = DataLoader(
            components.dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.dataset.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=components.collate_fn,
            persistent_workers=self.config.dataset.num_workers > 0,
        )
        self.impl = SFTTrainWorkerImpl(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            preprocessor=components.preprocessor,
            dataloader=dataloader,
            config=self.config,
            sampler=sampler,
        )
        if self.resume is not None:
            self.impl.step = load_checkpoint(Path(self.resume), model, optimizer, scheduler)

    def train(self) -> dict[str, float | int]:
        if self.impl is None:
            raise RuntimeError("SFT train worker is not set up")
        return self.impl.train()

    def teardown(self) -> None:
        self.impl = None
        if dist.is_initialized():
            dist.destroy_process_group()

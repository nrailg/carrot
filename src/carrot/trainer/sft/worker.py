"""Per-GPU SFT worker and its testable loop implementation."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import wandb
from torch import nn
from torch.distributed.tensor import DTensor
from torch.utils.data import DataLoader, DistributedSampler

from carrot.distributed import Worker
from carrot.models.pi05 import build_pi05
from carrot.models.pi05.parallelize import Pi05Parallelizer
from carrot.parallel import parallelize_model
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
    """Build the warmup/cosine schedule used by the PI0.5 recipe."""
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
            "dataset_factory": config.dataset.factory,
            "dataset_factory_kwargs": config.dataset.factory_kwargs,
            "steps": config.steps,
            "micro_batch_size": config.micro_batch_size,
            "global_batch_size": config.global_batch_size,
            "gas": config.gas,
            "dp_size": config.dp_size,
            "num_nodes": config.num_nodes,
            "learning_rate": config.optimizer.learning_rate,
        },
    )


class SFTTrainWorkerImpl:
    """Implementation detail that keeps the worker training loop maintainable."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        loss_fn: Any,
        checkpoint_artifact_writer: Callable[[Path], None] | None,
        preprocessor: Any,
        dataloader: DataLoader,
        config: SFTConfig,
        sampler: DistributedSampler | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.checkpoint_artifact_writer = checkpoint_artifact_writer
        self.preprocessor = preprocessor
        self.dataloader = dataloader
        self.config = config
        self.sampler = sampler
        self.step = 0
        self.epoch = 0
        self._wandb = None

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
                f"micro_batch_size={self.config.micro_batch_size} "
                f"global_batch_size={self.config.global_batch_size} "
                f"gas={self.config.gas} world={world_size}",
                flush=True,
            )
        consumed_batches = self.step * self.config.gas
        self.epoch, batch_offset = divmod(consumed_batches, len(self.dataloader))
        if self.sampler is not None:
            self.sampler.set_epoch(self.epoch)
        iterator = iter(self.dataloader)
        for _ in range(batch_offset):
            next(iterator)
        try:
            return self._train_loop(iterator, mean_loss)
        finally:
            if self._wandb is not None:
                self._wandb.finish()
                self._wandb = None

    def _train_loop(self, iterator, mean_loss: float) -> dict[str, float | int]:
        while self.step < self.config.steps:
            accumulated_loss = 0.0
            for micro_step in range(self.config.gas):
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
                loss, _ = self.loss_fn(self.model, batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss at step {self.step}: {loss.item()}"
                    )
                scaled = loss / self.config.gas
                if self._is_rank_0() and self.step == 0 and micro_step == 0:
                    print(
                        f"first forward ok loss={loss.detach().float().item():.6f} "
                        f"dtype={loss.dtype}",
                        flush=True,
                    )
                scaled.backward()
                accumulated_loss += loss.detach().float().item()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.optimizer.max_grad_norm,
            )
            if isinstance(grad_norm, DTensor):
                grad_norm = grad_norm.full_tensor()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1

            mean_loss = accumulated_loss / self.config.gas
            if dist.is_initialized():
                value = torch.tensor(mean_loss, device=torch.cuda.current_device())
                dist.all_reduce(value)
                mean_loss = value.item() / dist.get_world_size()
            learning_rate = self.optimizer.param_groups[0]["lr"]
            if self._is_rank_0() and self.step % self.config.log_freq == 0:
                print(
                    f"step={self.step} loss={mean_loss:.6f} lr={learning_rate:.3e} "
                    f"grad_norm={grad_norm.item():.6f}",
                    flush=True,
                )
                if self._wandb is not None:
                    self._wandb.log(
                        {
                            "train/loss": mean_loss,
                            "train/lr": learning_rate,
                            "train/grad_norm": grad_norm.item(),
                        },
                        step=self.step,
                    )
            if self.config.save_freq and self.step % self.config.save_freq == 0:
                save_checkpoint(
                    Path(self.config.output_dir) / "checkpoints" / f"step-{self.step:08d}",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.step,
                    self.checkpoint_artifact_writer,
                )
        if self.config.save_freq and self.step % self.config.save_freq != 0:
            save_checkpoint(
                Path(self.config.output_dir) / "checkpoints" / f"step-{self.step:08d}",
                self.model,
                self.optimizer,
                self.scheduler,
                self.step,
                self.checkpoint_artifact_writer,
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
        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend="nccl", rank=self.rank, world_size=self.world_size)
        random.seed(self.config.seed + self.rank)
        torch.manual_seed(self.config.seed + self.rank)
        torch.cuda.manual_seed_all(self.config.seed + self.rank)

        model_path = self.resume if self.resume is not None else self.config.model.path
        components = build_pi05(
            model_path=model_path,
            tokenizer_path=self.config.model.tokenizer_path,
            dataset_factory=self.config.dataset.factory,
            dataset_factory_kwargs=self.config.dataset.factory_kwargs,
            device=f"cuda:{self.local_rank}",
            norm_stats_path=self.config.dataset.norm_stats_path,
            preprocess=self.config.dataset.preprocess,
        )
        model = components.policy
        parallelize_model(model, Pi05Parallelizer(), self.config.fsdp)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.config.optimizer.learning_rate,
            weight_decay=self.config.optimizer.weight_decay,
            betas=self.config.optimizer.betas,
            eps=self.config.optimizer.eps,
        )
        scheduler = _scheduler(
            optimizer,
            self.config.optimizer.warmup_steps,
            self.config.steps,
            decay_steps=self.config.optimizer.decay_steps,
            decay_learning_rate=self.config.optimizer.decay_learning_rate,
        )
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
            batch_size=self.config.micro_batch_size,
            sampler=sampler,
            num_workers=self.config.dataset.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=components.collate_fn,
            persistent_workers=self.config.dataset.num_workers > 0,
            multiprocessing_context=(
                "forkserver" if self.config.dataset.num_workers > 0 else None
            ),
        )
        self.impl = SFTTrainWorkerImpl(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=components.loss_fn,
            checkpoint_artifact_writer=components.loss_fn.save_artifacts,
            preprocessor=lambda batch: batch,
            dataloader=dataloader,
            config=self.config,
            sampler=sampler,
        )
        if self.resume is not None:
            self.impl.step = load_checkpoint(Path(self.resume), optimizer, scheduler)

    def train(self) -> dict[str, float | int]:
        if self.impl is None:
            raise RuntimeError("SFT train worker is not set up")
        return self.impl.train()

    def teardown(self) -> None:
        self.impl = None
        if dist.is_initialized():
            dist.destroy_process_group()

"""Synchronous SFT loop and distributed worker."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from carrot.distributed import Worker
from carrot.modeling import parallelize_model
from carrot.models.smolvla import SmolVLAParallelizer, build_smolvla
from carrot.trainer.sft.checkpoint import load_checkpoint, save_checkpoint
from carrot.trainer.sft.config import SFTConfig


def _scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


class SFTTrainer:
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

    def train(self) -> dict[str, float | int]:
        self.model.train()
        iterator = iter(self.dataloader)
        mean_loss = float("nan")
        self.optimizer.zero_grad(set_to_none=True)
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
                sync = micro_step + 1 == self.config.gradient_accumulation_steps
                set_sync = getattr(self.model, "set_requires_gradient_sync", None)
                if callable(set_sync):
                    set_sync(sync)
                batch = self.preprocessor(batch)
                loss, _ = self.model(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss at step {self.step}: {loss.item()}"
                    )
                (loss / self.config.gradient_accumulation_steps).backward()
                accumulated_loss += loss.detach().float().item()

            if self.config.optimizer.max_grad_norm:
                clip = getattr(self.model, "clip_grad_norm_", None)
                if callable(clip):
                    clip(self.config.optimizer.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.optimizer.max_grad_norm
                    )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1

            mean_loss = accumulated_loss / self.config.gradient_accumulation_steps
            if dist.is_initialized():
                value = torch.tensor(mean_loss, device=torch.cuda.current_device())
                dist.all_reduce(value)
                mean_loss = value.item() / dist.get_world_size()
            if self._is_main() and self.step % self.config.log_freq == 0:
                print(
                    f"step={self.step} loss={mean_loss:.6f} "
                    f"lr={self.optimizer.param_groups[0]['lr']:.3e}",
                    flush=True,
                )
            if self.config.save_freq and self.step % self.config.save_freq == 0:
                save_checkpoint(
                    Path(self.config.output_dir) / "checkpoints" / f"step-{self.step:08d}",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.step,
                )
        return {"step": self.step, "loss": mean_loss}

    @staticmethod
    def _is_main() -> bool:
        return not dist.is_initialized() or dist.get_rank() == 0


class SFTTrainerWorker(Worker):
    def __init__(self, config: SFTConfig, resume: str | None = None) -> None:
        super().__init__()
        self.config = config
        self.resume = resume
        self.trainer: SFTTrainer | None = None

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SmolVLA FSDP training requires CUDA")
        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend="nccl", rank=self.rank, world_size=self.world_size)
        random.seed(self.config.seed + self.rank)
        torch.manual_seed(self.config.seed + self.rank)
        torch.cuda.manual_seed_all(self.config.seed + self.rank)

        components = build_smolvla(
            model_path=self.config.model.path,
            dataset_repo_id=self.config.dataset.repo_id,
            dataset_root=self.config.dataset.root,
            device=f"cuda:{self.local_rank}",
            video_backend=self.config.dataset.video_backend,
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
        )
        scheduler = _scheduler(
            optimizer,
            self.config.optimizer.warmup_steps,
            self.config.steps,
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
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.dataset.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=components.collate_fn,
            persistent_workers=self.config.dataset.num_workers > 0,
        )
        self.trainer = SFTTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            preprocessor=components.preprocessor,
            dataloader=dataloader,
            config=self.config,
            sampler=sampler,
        )
        if self.resume is not None:
            self.trainer.step = load_checkpoint(Path(self.resume), model, optimizer, scheduler)

    def train(self) -> dict[str, float | int]:
        if self.trainer is None:
            raise RuntimeError("trainer worker is not set up")
        return self.trainer.train()

    def teardown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

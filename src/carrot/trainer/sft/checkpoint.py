"""Distributed checkpoints for FSDP2 SFT."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import Tensor, nn
from torch.distributed.tensor import DTensor


class _OptimizerState:
    def __init__(self, optimizer: Any) -> None:
        self.optimizer = optimizer

    def state_dict(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)


def _initialize_optimizer_state(optimizer: Any) -> None:
    if optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for parameter in param_group["params"]:
            if parameter.grad is not None:
                raise RuntimeError("optimizer state must be restored before backward")
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)
    learning_rates = [param_group["lr"] for param_group in optimizer.param_groups]
    for param_group in optimizer.param_groups:
        param_group["lr"] = 0.0
    try:
        optimizer.step()
    finally:
        for param_group, learning_rate in zip(
            optimizer.param_groups,
            learning_rates,
            strict=True,
        ):
            param_group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)


def _get_full_model_state_dict(model: nn.Module) -> dict[str, Tensor]:
    rank_zero = not dist.is_initialized() or dist.get_rank() == 0
    full_state_dict = {}
    for name, value in model.state_dict().items():
        tensor = value.full_tensor() if isinstance(value, DTensor) else value
        if rank_zero:
            full_state_dict[name] = tensor.detach().cpu()
    return full_state_dict


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Any,
    scheduler: Any,
    step: int,
    artifact_writer: Callable[[Path], None] | None = None,
) -> None:
    """Write an OpenPI PyTorch model export and an optimizer-only DCP bundle.

    DCP and the full-state gather are collective; every rank must enter. Rank 0
    writes the model artifacts and atomically publishes the completed directory.

    Parameters
    ----------
    path : Path
        Checkpoint directory. ``from_pretrained`` reads the root OpenPI export;
        ``--resume`` additionally loads ``optimizer/`` and ``trainer_state.json``.
    model : nn.Module
        Must implement ``save_pretrained(save_directory, *, state_dict=...)``.
    optimizer : Any
    scheduler : Any
    step : int
    artifact_writer : collections.abc.Callable[[Path], None] | None
        Invoked on rank 0 after the OpenPI model export.
    """
    temporary_path = path.with_name(f"tmp-{path.name}")
    if path.exists():
        raise FileExistsError(f"checkpoint already exists: {path}")
    if not dist.is_initialized() or dist.get_rank() == 0:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        temporary_path.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    dcp.save(
        {"optimizer": _OptimizerState(optimizer)},
        checkpoint_id=str(temporary_path / "optimizer"),
    )
    # Direct DTensor gather keeps later optimizer-only DCP saves reusable under FSDP2.
    state_dict = _get_full_model_state_dict(model)
    if not dist.is_initialized() or dist.get_rank() == 0:
        model.save_pretrained(temporary_path, state_dict=state_dict)
        if artifact_writer is not None:
            artifact_writer(temporary_path)
        with (temporary_path / "trainer_state.json").open("w") as stream:
            json.dump({"step": step, "scheduler": scheduler.state_dict()}, stream)
    if dist.is_initialized():
        dist.barrier()
    if not dist.is_initialized() or dist.get_rank() == 0:
        temporary_path.rename(path)
    if dist.is_initialized():
        dist.barrier()


def load_checkpoint(
    path: Path,
    optimizer: Any,
    scheduler: Any,
) -> int:
    _initialize_optimizer_state(optimizer)
    state = _OptimizerState(optimizer)
    dcp.load({"optimizer": state}, checkpoint_id=str(path / "optimizer"))
    with (path / "trainer_state.json").open() as stream:
        trainer_state = json.load(stream)
    scheduler.load_state_dict(trainer_state["scheduler"])
    return int(trainer_state["step"])

"""Distributed checkpoints for FSDP2 SFT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_state_dict,
)


class _TrainState:
    def __init__(self, model: nn.Module, optimizer: Any) -> None:
        self.model = model
        self.optimizer = optimizer

    def state_dict(self) -> dict[str, Any]:
        model_state, optimizer_state = get_state_dict(self.model, self.optimizer)
        return {"model": model_state, "optimizer": optimizer_state}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
        )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Any,
    scheduler: Any,
    step: int,
) -> None:
    """Write a DCP resume bundle and a HuggingFace ``pretrained_model/`` export.

    ``dcp.save`` and the full-state gather are collective; every rank must enter.
    Rank 0 writes ``pretrained_model/`` and ``trainer_state.json``.

    Parameters
    ----------
    path : Path
        Checkpoint directory. ``--resume`` loads ``dcp/`` and
        ``trainer_state.json`` from here; ``from_pretrained`` uses
        ``path / "pretrained_model"``.
    model : nn.Module
        Must implement ``save_pretrained(save_directory, *, state_dict=...)``.
    optimizer : Any
    scheduler : Any
    step : int
    """
    if not dist.is_initialized() or dist.get_rank() == 0:
        path.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    dcp.save({"train": _TrainState(model, optimizer)}, checkpoint_id=str(path / "dcp"))
    # full_state_dict gather is collective; only rank 0 writes HF files.
    state_dict = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    if not dist.is_initialized() or dist.get_rank() == 0:
        model.save_pretrained(path / "pretrained_model", state_dict=state_dict)
        with (path / "trainer_state.json").open("w") as stream:
            json.dump({"step": step, "scheduler": scheduler.state_dict()}, stream)
    if dist.is_initialized():
        dist.barrier()


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Any,
    scheduler: Any,
) -> int:
    state = _TrainState(model, optimizer)
    dcp.load({"train": state}, checkpoint_id=str(path / "dcp"))
    with (path / "trainer_state.json").open() as stream:
        trainer_state = json.load(stream)
    scheduler.load_state_dict(trainer_state["scheduler"])
    return int(trainer_state["step"])

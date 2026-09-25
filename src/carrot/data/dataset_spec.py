"""Dataset contract used by PI0.5 offline SFT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SFTDatasetSpec:
    """Describe one PI0.5 SFT dataset integration.

    Parameters
    ----------
    dataset : Any
    collate_fn : Any
    state_stats : dict[str, Any]
    action_stats : dict[str, Any]
    image_keys : tuple[str, str, str]
    state_key : str
    action_key : str
    task_key : str
    embodiment : str
        ``libero`` and ``so101`` select shared input-transform pipelines.
    """

    dataset: Any
    collate_fn: Any
    state_stats: dict[str, Any]
    action_stats: dict[str, Any]
    image_keys: tuple[str, ...]
    state_key: str = "observation.state"
    action_key: str = "action"
    task_key: str = "task"
    embodiment: str = "robotwin"
